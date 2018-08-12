import os
import random

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
from toolz.sandbox import unzip
from torch.utils.data import Dataset
import more_itertools

from common.graph_embedding import GGNNLayer, MultiIterationGraph
from common import torch_util, util
from common.logger import info
from common.problem_util import to_cuda
from common.torch_util import create_sequence_length_mask, Update, MaskOutput, expand_tensor_sequence_list_to_same, \
    SequenceMaskOutput, expand_tensor_sequence_len, expand_tensor_sequence_to_same, DynamicDecoder, \
    pad_last_dim_of_tensor_list, BeamSearchDynamicDecoder
from common.util import PaddedList, create_effect_keyword_ids_set
from seq2seq.models import EncoderRNN, DecoderRNN

"""
The batch_data dict should has the following keys:
'p1_target': , 
'p2_target', 'is_copy_target', 'copy_target', 'sample_target', 'adj', 'input_seq', 'input_length', 
'copy_length', 'target', 
'compatible_tokens',
'compatible_tokens_length'
"""


class SliceEncoder(nn.Module):
    def __init__(self, rnn_type, hidden_size, n_layer, dropout_p, inner=False):
        """
        This module extract a slice from a sequence and encode the slice into a state vector
        """
        super().__init__()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=n_layer,
                               batch_first=True, bidirectional=True, dropout=dropout_p)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=n_layer,
                              batch_first=True, bidirectional=True, dropout=dropout_p)
        self.inner = inner
        if not inner:
            self.transform = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, p1, p2, seq, copy_length):
        """
        :param p1: The slice begin position, shape [batch, ]
        :param p2: The slice end position, shape [batch, ]
        :param seq: The sequence shape [batch, seq, dim]
        :param copy_length: The copy_length shape [batch,]
        :return: The encoded slice vector, shape [batch, ]
        """
        seq = torch.unbind(seq, dim=0)
        copy_length = torch.unbind(copy_length, dim=0)
        if self.inner:
            seq = [s[a:b+1] for a, b, s in zip(p1, p2, seq)]
            return self._encode(seq)
        else:
            seq_before = [s[:a+1] for a, s in zip(p1, seq)]
            before_encoder = self._encode(seq_before)
            seq_after = [s[a:e] for a, s, e in zip(p2, seq, copy_length)]
            # seq_after = [s[a:] for a, s in zip(p2, seq)]
            after_encoder = self._encode(seq_after)
            encoder = torch.cat((before_encoder, after_encoder), dim=-1)
            return self.transform(encoder)

    def _encode(self, seq):
        packed_seq, idx_unsort = torch_util.pack_sequence(seq)
        # self.rnn.flatten_parameters()
        _, encoded_state = self.rnn(packed_seq)
        return torch.index_select(encoded_state, 1, idx_unsort.to(seq[0].device))


class PointerNetwork(nn.Module):
    def __init__(self, hidden_size, use_query_vector=False):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        if use_query_vector:
            self.query_transform = nn.Linear(hidden_size, hidden_size)
        self.query_vector = nn.Parameter(torch.randn(1, hidden_size, 1))

    def forward(self, x, query=None, mask=None):
        """
        :param x: shape [batch, seq, dim]
        :param query: shape [dim]
        :param mask: shape [batch, seq]
        :return: shape [batch, seq]
        """
        batch_size = x.shape[0]
        x = self.transform(x)
        if query is not None:
            x = x + self.query_transform(query)
        x = F.tanh(x)
        x = torch.bmm(x, self.query_vector.expand(batch_size, -1, -1))
        x = x.squeeze(-1)
        if mask is not None:
            x.data.masked_fill_(~mask, -float('inf'))
        return x


class RNNGraphWrapper(nn.Module):
    def __init__(self, hidden_size, parameter):
        super().__init__()
        self.encoder = EncoderRNN(hidden_size=hidden_size, **parameter)
        self.bi = 2 if parameter['bidirectional'] else 1
        self.transform_size = nn.Linear(self.bi * hidden_size, hidden_size)

    def forward(self, x, adj, copy_length):
        o, _ = self.encoder(x)
        o = self.transform_size(o)
        return o


class MixedRNNGraphWrapper(nn.Module):
    def __init__(self,
                 hidden_size,
                 rnn_parameter,
                 graph_type,
                 graph_itr,
                 dropout_p=0,
                 mask_ast_node_in_rnn=False,
                 ):
        super().__init__()
        self.rnn = nn.ModuleList([RNNGraphWrapper(hidden_size, rnn_parameter) for _ in range(graph_itr)])
        self.graph_itr = graph_itr
        self.dropout = nn.Dropout(dropout_p)
        self.mask_ast_node_in_rnn = mask_ast_node_in_rnn
        self.inner_graph_itr = 1
        if graph_type == 'ggnn':
            self.graph = GGNNLayer(hidden_size)

    def forward(self, x, adj, copy_length):
        if self.mask_ast_node_in_rnn:
            copy_length_mask = create_sequence_length_mask(copy_length, x.shape[1]).unsqueeze(-1)
            zero_fill = torch.zeros_like(x)
            for i in range(self.graph_itr):
                tx = torch.where(copy_length_mask, x, zero_fill)
                tx = tx + self.rnn[i](tx, adj, copy_length)
                x = torch.where(copy_length_mask, tx, x)
                x = self.dropout(x)
                # for _ in range(self.inner_graph_itr):
                x = x + self.graph(x, adj)
                if i < self.graph_itr - 1:
                    # pass
                    x = self.dropout(x)
        else:
            for i in range(self.graph_itr):
                x = x + self.rnn[i](x, adj, copy_length)
                x = self.dropout(x)
                x = x + self.graph(x, adj)
                if i < self.graph_itr - 1:
                    x = self.dropout(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self,
                 hidden_size=300,
                 graph_embedding='ggnn',
                 graph_parameter={},
                 pointer_type='itr',
                 embedding=None, embedding_size=400,
                 ):
        """
        :param hidden_size: The hidden state size in the model
        :param graph_embedding: The graph propagate method, option:["ggnn", "graph_attention", "rnn"]
        :param pointer_type: The method to point out the begin and the end, option:["itr", "query"]
        """
        super().__init__()
        self.pointer_type = pointer_type
        self.embedding = embedding
        self.size_transform = nn.Linear(embedding_size, hidden_size)
        if self.pointer_type == 'itr':
            self.pointer_transform = nn.Linear(2 * hidden_size, 1)
        elif self.pointer_type == 'query':
            self.query_tensor = nn.Parameter(torch.randn(hidden_size))
            self.up_data_cell = nn.GRUCell(hidden_size, hidden_size, bias=True)
            self.p1_pointer_network = PointerNetwork(hidden_size, use_query_vector=True)
            self.p2_pointer_network = PointerNetwork(hidden_size, use_query_vector=True)
        self.graph_embedding = graph_embedding
        if graph_embedding == 'ggnn':
            self.graph = MultiIterationGraph(GGNNLayer(hidden_state_size=hidden_size), **graph_parameter)
        elif graph_embedding == 'rnn':
            self.graph = RNNGraphWrapper(hidden_size=hidden_size, parameter=graph_parameter)
        elif graph_embedding == 'mixed':
            self.graph = MixedRNNGraphWrapper(hidden_size, **graph_parameter)

    def forward(self,
                adjacent_matrix,
                input_seq,
                copy_length,
                ):
        input_seq = self.embedding(input_seq)
        input_seq = self.size_transform(input_seq)
        input_seq = self.graph(input_seq, adjacent_matrix, copy_length)
        batch_size = input_seq.shape[0]
        pointer_mask = torch_util.create_sequence_length_mask(copy_length, max_len=input_seq.shape[1])
        if self.pointer_type == 'itr':
            input_seq0 = input_seq
            input_seq1 = self.graph(input_seq, adjacent_matrix, copy_length)
            input_seq2 = self.graph(input_seq1, adjacent_matrix, copy_length)
            p1 = self.pointer_transform(torch.cat((input_seq0, input_seq1), dim=-1), ).squeeze(-1)
            p1.data.masked_fill_(~pointer_mask, -float("inf"))
            p2 = self.pointer_transform(torch.cat((input_seq0, input_seq2), dim=-1), ).squeeze(-1)
            p2.data.masked_fill_(~pointer_mask, -float("inf"))
            input_seq = self.graph(input_seq2, adjacent_matrix, copy_length)
        elif self.pointer_type == 'query':
            p1 = self.p1_pointer_network(input_seq, query=self.query_tensor, mask=pointer_mask)
            p1_state = torch.sum(F.softmax(p1, dim=-1).unsqueeze(-1) * input_seq, dim=1)
            p2_query = self.up_data_cell(p1_state, self.query_tensor.unsqueeze(0).expand(batch_size, -1))
            p2 = self.p2_pointer_network(input_seq, query=torch.unsqueeze(p2_query, dim=1), mask=pointer_mask)
        else:
            raise ValueError("No point type is:{}".format(self.pointer_type))

        return p1, p2, input_seq


class Output(nn.Module):
    def __init__(self, hidden_size, vocabulary_size):
        super().__init__()
        self.is_copy_output = nn.Linear(hidden_size, 1)
        self.copy_embedding = nn.Embedding(2, hidden_size)
        self.update = Update(hidden_size)
        self.sample_output = SequenceMaskOutput(hidden_size, vocabulary_size)

    def forward(self, decoder_output, encoder_output, copy_mask, sample_mask, sample_length, is_sample=False,
                copy_target=None):
        is_copy = self.is_copy_output(decoder_output).squeeze(-1)
        if is_sample:
            copy_target = (is_copy > 0.5).long()
        else:
            copy_target = (copy_target == 1).long()

        copy_state = self.copy_embedding(copy_target)
        decoder_output = self.update(decoder_output, copy_state)
        copy_output = torch.bmm(decoder_output, encoder_output.permute(0, 2, 1))
        copy_output.data.masked_fill_(~torch.unsqueeze(copy_mask, dim=1), -float('inf'))
        sample_length_shape = sample_length.shape
        sample_length_mask = create_sequence_length_mask(sample_length.view(-1, ), max_len=sample_mask.shape[-1]).view(*list(sample_length_shape), -1)
        sample_output = self.sample_output(decoder_output, sample_mask, sample_length_mask)
        return is_copy, copy_output, sample_output


class EncoderSampleModel(nn.Module):
    def __init__(self,
                 start_label,
                 end_label,
                 inner_start_label,
                 inner_end_label,
                 vocabulary_size,
                 embedding_size=300,
                 hidden_size=300,
                 max_sample_length=10,
                 graph_parameter={},
                 graph_embedding='ggnn',
                 pointer_type='itr',
                 rnn_type='gru',
                 rnn_layer_number=3,
                 max_length=500,
                 dropout_p=0.1,
                 pad_label=-1,
                 vocabulary=None,
                 mask_type='static',
                 beam_size=5
                 ):
        """
        :param vocabulary_size: The size of token vocabulary
        :param embedding_size: The embedding size
        :param hidden_size: The hidden state size in the model
        :param max_sample_length: The max length to sample
        :param graph_embedding: The graph propagate method, option:["ggnn", "graph_attention", "rnn", "mixed]
        :param pointer_type: The method to point out the begin and the end, option:["itr", "query"]
        :param rnn_type: The rnn type used in the model, option:["lstm", "gru"]
        :param rnn_layer_number: The number of layer in this model
        :param dropout_p: The dropout p in this model
        """
        super().__init__()
        self.max_sample_length = max_sample_length
        self.inner_start_label = inner_start_label
        self.inner_end_label = inner_end_label
        self.pad_label = pad_label
        self.beam_size = beam_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.graph_encoder = GraphEncoder(hidden_size=hidden_size, graph_embedding=graph_embedding,
                                          pointer_type=pointer_type, graph_parameter=graph_parameter,
                                          embedding=self.embedding, embedding_size=embedding_size)
        self.slice_encoder = SliceEncoder(rnn_type=rnn_type, hidden_size=hidden_size // 2, n_layer=rnn_layer_number,
                                          dropout_p=dropout_p)
        # self.slice_encoder = SliceEncoder(rnn_type=rnn_type, hidden_size=hidden_size, n_layer=rnn_layer_number,
        #                                   dropout_p=dropout_p)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=max_sample_length, hidden_size=hidden_size,
                                  sos_id=inner_start_label, eos_id=inner_end_label, n_layers=rnn_layer_number, rnn_cell=rnn_type,
                                  bidirectional=False, input_dropout_p=dropout_p, dropout_p=dropout_p,
                                  use_attention=True)
        self.output = Output(hidden_size, vocabulary_size)
        self.layer_number = rnn_layer_number
        self.hidden_size = hidden_size

        self.dynamic_decoder = DynamicDecoder(self.inner_start_label, self.inner_end_label, self.pad_label,
                                              self.decoder_one_step, self.create_next_output_input,
                                              self.max_sample_length)
        self.beam_dynamic_decoder = BeamSearchDynamicDecoder(self.inner_start_label, self.inner_end_label, self.pad_label,
                                              self.decoder_one_step, self.beam_create_next_output_input,
                                              self.max_sample_length, self.beam_size)
        self.vocabulary = vocabulary
        self.keyword_ids_set = create_effect_keyword_ids_set(vocabulary)
        self.mask_type = mask_type

    def _train_forward(self,
                       adjacent_matrix,
                       input_seq,
                       input_length,
                       copy_length,
                       p1_target,
                       p2_target,
                       target,
                       compatible_tokens,
                       compatible_tokens_length,
                       is_copy_target,
                       ):
        batch_size = input_seq.shape[0]
        p1_o, p2_o, input_seq = self.graph_encoder(adjacent_matrix, input_seq, copy_length)
        slice_state = self.slice_encoder(p1_target, p2_target, input_seq, copy_length)\
            .permute(1, 0, 2)\
            .contiguous()\
            .view(batch_size, self.layer_number, -1)\
            .permute(1, 0, 2)\
            .contiguous()
        encoder_mask = create_sequence_length_mask(input_length, max_len=input_seq.shape[1])
        # do decoder and calculate output
        decoder_output, _, _ = self.decoder(inputs=self.embedding(target), encoder_hidden=slice_state,
                                            encoder_outputs=input_seq, encoder_mask=~encoder_mask,
                                            teacher_forcing_ratio=1)
        decoder_output = torch.stack(decoder_output, dim=1)
        copy_mask = create_sequence_length_mask(copy_length, max_len=input_seq.shape[1])
        is_copy, copy_output, sample_output = self.output(decoder_output, input_seq, copy_mask, compatible_tokens,
                                                          compatible_tokens_length, is_sample=False,
                                                          copy_target=is_copy_target)
        return p1_o, p2_o, is_copy, copy_output, sample_output, compatible_tokens

    def _sample_forward(self,
                        adjacent_matrix,
                        ori_input_seq,
                        input_length,
                        copy_length,
                        ):
        batch_size = ori_input_seq.shape[0]
        p1_o, p2_o, input_seq = self.graph_encoder(adjacent_matrix, ori_input_seq, copy_length)
        p1 = torch.squeeze(torch.topk(F.log_softmax(p1_o, dim=-1), k=1, dim=-1)[1], dim=-1)
        p2 = torch.squeeze(torch.topk(F.log_softmax(p2_o, dim=-1), k=1, dim=-1)[1], dim=-1)
        slice_state = self.slice_encoder(p1, p2, input_seq, copy_length) \
            .permute(1, 0, 2) \
            .contiguous() \
            .view(batch_size, self.layer_number, -1) \
            .permute(1, 0, 2) \
            .contiguous()
        encoder_mask = create_sequence_length_mask(input_length, )

        self.compatible_tokens = None
        self.compatible_tokens_length = None

        decoder_output_list, _, _ = self.dynamic_decoder.decoder(encoder_output=input_seq, endocer_hidden=slice_state, encoder_mask=encoder_mask,
                                     copy_length=copy_length, ori_input_seq=ori_input_seq, input_length=input_length)
        is_copy_list, copy_output_list, sample_output_list, compatible_tokens_list = list(zip(*decoder_output_list))

        is_copy = torch.cat(is_copy_list, dim=1)
        copy_output = torch.cat(copy_output_list, dim=1)
        padded_sample_output_list = pad_last_dim_of_tensor_list(sample_output_list, fill_value=-float('inf'))
        sample_output = torch.cat(padded_sample_output_list, dim=1)
        padded_compatible_tokens_list = pad_last_dim_of_tensor_list(compatible_tokens_list, fill_value=0)
        pad_compatible_tokens = torch.cat(padded_compatible_tokens_list, dim=1)
        return p1_o, p2_o, is_copy, copy_output, sample_output, pad_compatible_tokens

    def _beam_search_sample_forward(self,
                        adjacent_matrix,
                        ori_input_seq,
                        input_length,
                        copy_length,
                        ):
        batch_size = ori_input_seq.shape[0]
        p1_o, p2_o, input_seq = self.graph_encoder(adjacent_matrix, ori_input_seq, copy_length)
        p1 = torch.squeeze(torch.topk(F.log_softmax(p1_o, dim=-1), k=1, dim=-1)[1], dim=-1)
        p2 = torch.squeeze(torch.topk(F.log_softmax(p2_o, dim=-1), k=1, dim=-1)[1], dim=-1)
        slice_state = self.slice_encoder(p1, p2, input_seq) \
            .permute(1, 0, 2) \
            .contiguous() \
            .view(batch_size, self.layer_number, -1) \
            .permute(1, 0, 2) \
            .contiguous()
        encoder_mask = create_sequence_length_mask(input_length, )

        decoder_output_list, _, _ = self.beam_dynamic_decoder.decoder(encoder_output=input_seq, endocer_hidden=slice_state, encoder_mask=encoder_mask,
                                     copy_length=copy_length, ori_input_seq=ori_input_seq, input_length=input_length)
        is_copy_list, copy_output_list, sample_output_list, compatible_tokens_list = list(zip(*decoder_output_list))

        is_copy = torch.cat(is_copy_list, dim=2)
        copy_output = torch.cat(copy_output_list, dim=2)
        # padded_sample_output_list = pad_last_dim_of_tensor_list(sample_output_list, fill_value=0)
        padded_sample_output_list = sample_output_list
        sample_output = torch.cat(padded_sample_output_list, dim=2)
        # padded_compatible_tokens_list = pad_last_dim_of_tensor_list(compatible_tokens_list, fill_value=0)
        padded_compatible_tokens_list = compatible_tokens_list
        pad_compatible_tokens = torch.cat(padded_compatible_tokens_list, dim=2)

        # get beam 0 result
        is_copy = is_copy[:, 0]
        copy_output = copy_output[:, 0]
        sample_output = sample_output[:, 0]
        pad_compatible_tokens = pad_compatible_tokens[:, 0]
        return p1_o, p2_o, is_copy, copy_output, sample_output, pad_compatible_tokens

    def create_one_step_token_masks(self, ori_input_seq, input_length, continue_mask):
        if self.mask_type == 'static':
            if not hasattr(self, 'compatible_tokens') or self.compatible_tokens is None:
                self.compatible_tokens, self.compatible_tokens_length = self.create_static_token_mask(ori_input_seq, input_length)
            return self.compatible_tokens, self.compatible_tokens_length
        return None, None

    def create_static_token_mask(self, ori_input_seq, input_length):
        ori_input_seq_list = ori_input_seq.tolist()
        input_length_list = input_length.tolist()
        mask_list = []
        for inp, l in zip(ori_input_seq_list, input_length_list):
            input_set = self.keyword_ids_set | set(inp[1:l-1]) | {self.inner_end_label}
            mask_list += [[sorted(input_set)]]
        mask_len_list = [[len(j) for j in i] for i in mask_list]
        compatible_tokens = torch.LongTensor(PaddedList(mask_list)).to(ori_input_seq.device)
        compatible_tokens_length = torch.LongTensor(PaddedList(mask_len_list)).to(ori_input_seq.device)
        return compatible_tokens, compatible_tokens_length

    def decoder_one_step(self, decoder_inputs, continue_mask, start_index, hidden, encoder_output, encoder_mask, copy_length,
                         ori_input_seq=None, input_length=None):
        decoder_output, hidden, _ = self.decoder(inputs=self.embedding(decoder_inputs), encoder_hidden=hidden,
                                            encoder_outputs=encoder_output, encoder_mask=~encoder_mask,
                                            teacher_forcing_ratio=0)
        decoder_output = torch.stack(decoder_output, dim=1)
        copy_mask = create_sequence_length_mask(copy_length)

        compatible_tokens, compatible_tokens_length = \
            self.create_one_step_token_masks(ori_input_seq=ori_input_seq, input_length=input_length, continue_mask=continue_mask)

        is_copy, copy_output, sample_output = self.output(decoder_output, encoder_output, copy_mask, compatible_tokens,
                                                          compatible_tokens_length, is_sample=True,
                                                          copy_target=None)
        error_list = []
        return (is_copy, copy_output, sample_output, compatible_tokens), hidden, error_list

    def create_next_output_input(self, one_step_decoder_output, ori_input_seq=None, **kwargs):
        is_copy, copy_output, sample_output, compatible_tokens = one_step_decoder_output
        # is_copy = is_copy > 0.5
        is_copy = torch.sigmoid(is_copy) > 0.5
        copy_output_id = torch.squeeze(torch.topk(F.log_softmax(copy_output, dim=-1), dim=-1, k=1)[1], dim=-1)
        sample_output_id = torch.topk(F.log_softmax(sample_output, dim=-1), dim=-1, k=1)[1]
        sample_output = torch.squeeze(torch.gather(compatible_tokens, dim=-1, index=sample_output_id), dim=-1)

        input_seq = ori_input_seq
        copy_ids = torch.gather(input_seq, index=copy_output_id, dim=-1)
        sample_output_ids = torch.where(is_copy, copy_ids, sample_output)
        return sample_output_ids

    def beam_create_next_output_input(self, one_step_decoder_output, beam_size, continue_mask=None, ori_input_seq=None, **kwargs):
        is_copy, copy_output, sample_output, compatible_tokens = one_step_decoder_output
        is_copy = torch.sigmoid(is_copy)
        # do not beam search on is_copy
        is_copy_probs = torch.where(is_copy > 0.5, is_copy, 1.0 - is_copy)
        is_copy_probs = torch.unsqueeze(torch.log(is_copy_probs), dim=1).expand(-1, beam_size, *[-1 for _ in range(len(is_copy.shape)-1)])
        is_copy = is_copy > 0.5
        is_copy_res = torch.unsqueeze(is_copy, dim=1).expand(-1, beam_size, *[-1 for _ in range(len(is_copy.shape)-1)])

        # copy_ids: [batch, sample_seq, beam]
        copy_probs, copy_ids = torch.topk(F.log_softmax(copy_output, dim=-1), dim=-1, k=beam_size)
        copy_ids = torch.gather(torch.unsqueeze(ori_input_seq, dim=1), index=copy_ids, dim=-1)
        sample_output_probs, sample_output_id = torch.topk(F.log_softmax(sample_output, dim=-1),
                                                           dim=-1, k=beam_size)
        sample_output = torch.gather(compatible_tokens, dim=-1, index=sample_output_id)

        # it is a simple way to calculate beam when seq=1, if seq >1, error.
        copy_probs = copy_probs.permute(0, 2, 1)
        copy_ids = copy_ids.permute(0, 2, 1)
        sample_output_probs = sample_output_probs.permute(0, 2, 1)
        sample_output = sample_output.permute(0, 2, 1)

        copy_total_probs = is_copy_probs + copy_probs
        sample_total_probs = is_copy_probs + sample_output_probs
        total_probs = torch.where(is_copy_res, copy_total_probs, sample_total_probs)

        total_probs = torch.where(continue_mask.view(total_probs.shape[0], *[1 for _ in range(len(total_probs.shape)-1)]),
                                  total_probs, torch.zeros_like(total_probs))

        sample_output_ids = torch.where(is_copy_res, copy_ids, sample_output)
        beam_compatible_tokens = torch.unsqueeze(compatible_tokens, dim=1).expand(-1, beam_size, *[-1 for _ in range(len(compatible_tokens.shape)-1)])
        return sample_output_ids, total_probs, (is_copy_res, copy_ids, sample_output, beam_compatible_tokens)

    def forward(self,
                adjacent_matrix,
                input_seq,
                input_length,
                copy_length,
                p1_target=None,
                p2_target=None,
                target=None,
                is_copy_target=None,
                compatible_tokens=None,
                compatible_tokens_length=None,
                do_sample=False,
                do_beam_search=False,
                ):
        if do_beam_search:
            return self._beam_search_sample_forward(adjacent_matrix,
                                        input_seq,
                                        input_length,
                                        copy_length)
        if do_sample:
            return self._sample_forward(adjacent_matrix,
                                        input_seq,
                                        input_length,
                                        copy_length)
        else:
            return self._train_forward(adjacent_matrix,
                                       input_seq,
                                       input_length,
                                       copy_length,
                                       p1_target,
                                       p2_target,
                                       target,
                                       compatible_tokens,
                                       compatible_tokens_length, is_copy_target)


def create_loss_fn(ignore_id):
    bce_loss = nn.BCEWithLogitsLoss(reduce=False)
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)

    def loss_fn(p1_o, p2_o, is_copy, copy_output, sample_output, output_compatible_tokens,
                p1_target, p2_target, is_copy_target, copy_target, sample_target, sample_small_target):
        try:
            is_copy_loss = torch.mean(bce_loss(is_copy, is_copy_target) * torch.ne(is_copy_target, ignore_id).float())
        except Exception as e:
            print('')
            raise Exception(e)
        p1_loss = cross_loss(p1_o, p1_target)
        p2_loss = cross_loss(p2_o, p2_target)
        copy_loss = cross_loss(copy_output.permute(0, 2, 1), copy_target)
        sample_loss = cross_loss(sample_output.permute(0, 2, 1), sample_small_target)
        return is_copy_loss + sample_loss + p1_loss + p2_loss + copy_loss
    return loss_fn


def create_parse_target_batch_data(ignore_token):
    def parse_target_batch_data(batch_data):
        p1 = to_cuda(torch.LongTensor(batch_data['p1_target']))
        p2 = to_cuda(torch.LongTensor(batch_data['p2_target']))
        is_copy = to_cuda(torch.FloatTensor(PaddedList(batch_data['is_copy_target'], fill_value=ignore_token)))
        copy_target = to_cuda(torch.LongTensor(PaddedList(batch_data['copy_target'], fill_value=ignore_token)))
        sample_target = to_cuda(torch.LongTensor(PaddedList(batch_data['sample_target'], fill_value=ignore_token)))
        sample_small_target = to_cuda(torch.LongTensor(PaddedList(batch_data['sample_small_target'])))
        return p1, p2, is_copy, copy_target, sample_target, sample_small_target

    return parse_target_batch_data


def create_parse_input_batch_data_fn(use_ast=False):
    def parse_input_batch_data_fn(batch_data, do_sample):
        def to_long(x):
            return to_cuda(torch.LongTensor(x))

        if not use_ast:
            adjacent_matrix = to_long(batch_data['adj'])
        else:
            adjacent_tuple = [[[i]+tt for tt in t] for i, t in enumerate(batch_data['adj'])]
            adjacent_tuple = [list(t) for t in unzip(more_itertools.flatten(adjacent_tuple))]
            size = max(batch_data['input_length'])
            # print("max length in this batch:{}".format(size))
            adjacent_tuple = torch.LongTensor(adjacent_tuple)
            adjacent_values = torch.ones(adjacent_tuple.shape[1]).long()
            adjacent_size = torch.Size([len(batch_data['input_length']), size, size])
            adjacent_matrix = to_cuda(
                torch.sparse.LongTensor(
                    adjacent_tuple,
                    adjacent_values,
                    adjacent_size,
                ).float().to_dense()
            )
        input_seq = to_long(PaddedList(batch_data['input_seq']))
        input_length = to_long(batch_data['input_length'])
        copy_length = to_long(batch_data['copy_length'])
        if not do_sample:
            p1_target = to_long(batch_data['p1_target'])
            p2_target = to_long(batch_data['p2_target'])
            target = to_long(PaddedList(batch_data['target']))
            is_copy_target = to_long(PaddedList(batch_data['is_copy_target']))
            batch_size = len(batch_data['is_copy_target'])
            seq_len = is_copy_target.shape[1]
            max_mask_len = max(max(batch_data['compatible_tokens_length']))
            compatible_tokens = to_long(PaddedList(batch_data['compatible_tokens'], shape=[batch_size, seq_len, max_mask_len]))
            compatible_tokens_length = to_long(PaddedList(batch_data['compatible_tokens_length']))
            return adjacent_matrix, input_seq, input_length, copy_length, p1_target, p2_target, target, \
                   is_copy_target, \
                   compatible_tokens, \
                   compatible_tokens_length
        else:
            return adjacent_matrix, input_seq, input_length, copy_length
    return parse_input_batch_data_fn


def create_output_ids_fn(end_id):
    def create_output_ids(model_output, model_input, do_sample=False):
        output_record_list = create_records_all_output(model_input=model_input, model_output=model_output,
                                                       do_sample=do_sample)
        p1, p2, is_copy, copy_ids, sample_output, sample_output_ids = output_record_list

        sample_output_ids_list = sample_output_ids.tolist()
        effect_sample_output_list = []
        for sample in sample_output_ids_list:
            try:
                end_pos = sample.index(end_id)
                sample = sample[:end_pos]
            except ValueError as e:
                pass
            effect_sample_output_list += [sample]

        input_seq = model_input[1]
        input_seq_len = model_input[2].tolist()
        input_seq_list = torch.unbind(input_seq, dim=0)
        final_output = []
        for i, one_input in enumerate(input_seq_list):
            effect_sample = effect_sample_output_list[i]
            input_len = input_seq_len[i]
            one_output = torch.cat([one_input[1:p1[i]+1], torch.LongTensor(effect_sample).to(one_input.device), one_input[p2[i]:input_len-1]], dim=-1)
            final_output += [one_output]
        # pad output tensor in python list final output
        final_output = expand_tensor_sequence_list_to_same(final_output, dim=0, fill_value=0)
        outputs = torch.stack(final_output, dim=0)
        return outputs, sample_output_ids
    return create_output_ids


def expand_output_and_target_fn(ignore_token):
    def expand_output_and_target(model_output, model_target):
        model_output = list(model_output)
        model_target = list(model_target)
        # p1_o, p2_o, is_copy, copy_output, sample_output = model_output
        # p1, p2, is_copy, copy_target, sample_target, sample_small_target = model_target
        model_output[2], model_target[2] = expand_tensor_sequence_to_same(model_output[2], model_target[2],
                                                                          fill_value=0)
        model_output[3], model_target[3] = expand_tensor_sequence_to_same(model_output[3], model_target[3],
                                                                          fill_value=ignore_token)
        model_output[4], model_target[4] = expand_tensor_sequence_to_same(model_output[4], model_target[4],
                                                                          fill_value=ignore_token)
        max_expand_len = model_output[4].shape[1]
        model_target[5] = expand_tensor_sequence_len(model_target[5], max_expand_len, fill_value=ignore_token)
        model_output[5] = expand_tensor_sequence_len(model_output[5], max_expand_len)
        return model_output, model_target
    return expand_output_and_target


def create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id):
    def create_multi_step_next_input_batch(input_data, model_input, model_output, continue_list, direct_output=False):
        # output_record_list = create_records_all_output(model_input=model_input, model_output=model_output, do_sample=True)
        output_record_list = create_records_all_output_for_beam(model_input=model_input, model_output=model_output,
                                                                do_sample=True, direct_output=direct_output)
        p1, p2, is_copy, copy_ids, sample_output, sample_output_ids = output_record_list

        sample_output_ids_list = sample_output_ids.tolist()
        effect_sample_output_list = []
        for sample in sample_output_ids_list:
            try:
                end_pos = sample.index(inner_end_id)
                sample = sample[:end_pos]
            except ValueError as e:
                pass
            effect_sample_output_list += [sample]

        input_seq = input_data['input_seq']
        final_output = []
        for i, one_input in enumerate(input_seq):
            effect_sample = effect_sample_output_list[i]
            one_output = one_input[1:p1[i] + 1] + effect_sample + one_input[p2[i]:-1]
            final_output += [one_output]

        next_input = [[begin_id] + one + [end_id] for one in final_output]
        next_input = [next_inp if con else ori_inp for ori_inp, next_inp, con in
                      zip(input_data['input_seq'], next_input, continue_list)]
        next_input_len = [len(one) for one in next_input]
        final_output = [next_inp[1:-1] for next_inp in next_input]

        input_data['input_seq'] = next_input
        input_data['input_length'] = next_input_len
        input_data['copy_length'] = next_input_len
        return input_data, final_output, output_record_list
    return create_multi_step_next_input_batch


def create_records_all_output(model_input, model_output, do_sample=False):
    p1_o, p2_o, is_copy, copy_output, sample_output, output_compatible_tokens = model_output
    if do_sample:
        compatible_tokens = output_compatible_tokens
    else:
        compatible_tokens = model_input[8]
    p1 = torch.squeeze(torch.topk(F.softmax(p1_o, dim=-1), dim=-1, k=1)[1], dim=-1)
    p2 = torch.squeeze(torch.topk(F.softmax(p2_o, dim=-1), dim=-1, k=1)[1], dim=-1)
    is_copy = torch.squeeze(torch.sigmoid(is_copy), dim=-1) > 0.5
    # is_copy = torch.squeeze(is_copy, dim=-1) > 0.5
    copy_output_id = torch.squeeze(torch.topk(F.softmax(copy_output, dim=-1), dim=-1, k=1)[1], dim=-1)
    sample_output_id = torch.topk(F.softmax(sample_output, dim=-1), dim=-1, k=1)[1]
    sample_output = torch.squeeze(torch.gather(compatible_tokens, dim=-1, index=sample_output_id), dim=-1)

    input_seq = model_input[1]
    copy_ids = torch.gather(input_seq, index=copy_output_id, dim=-1)
    sample_output_ids = torch.where(is_copy, copy_ids, sample_output)
    return p1, p2, is_copy, copy_ids, sample_output, sample_output_ids


def create_records_all_output_for_beam(model_input, model_output, do_sample=False, direct_output=False):
    p1_o, p2_o, is_copy, copy_output, sample_output, output_compatible_tokens = model_output
    if do_sample:
        compatible_tokens = output_compatible_tokens
    else:
        compatible_tokens = model_input[8]
    p1 = torch.squeeze(torch.topk(F.softmax(p1_o, dim=-1), dim=-1, k=1)[1], dim=-1)
    p2 = torch.squeeze(torch.topk(F.softmax(p2_o, dim=-1), dim=-1, k=1)[1], dim=-1)
    if not direct_output:
        is_copy = torch.squeeze(torch.sigmoid(is_copy), dim=-1) > 0.5
        copy_output_id = torch.squeeze(torch.topk(F.softmax(copy_output, dim=-1), dim=-1, k=1)[1], dim=-1)
        sample_output_id = torch.topk(F.softmax(sample_output, dim=-1), dim=-1, k=1)[1]
        sample_output = torch.squeeze(torch.gather(compatible_tokens, dim=-1, index=sample_output_id), dim=-1)

        input_seq = model_input[1]
        copy_output = torch.gather(input_seq, index=copy_output_id, dim=-1)
    sample_output_ids = torch.where(is_copy, copy_output, sample_output)
    return p1, p2, is_copy, copy_output, sample_output, sample_output_ids


def multi_step_print_output_records_fn(inner_end_id):

    def multi_step_print_output_records(output_records, final_output, batch_data, step_i, vocabulary, compile_result_list):
        # info('------------------------------- in step {} ------------------------------------'.format(step_i))
        id_to_word_fn = lambda x: [vocabulary.id_to_word(t) for t in x]
        id_to_code_fn = lambda x: ' '.join(id_to_word_fn(x))
        step_times = len(final_output)
        batch_size = len(final_output[0])
        output_records_list = [[p.tolist() for p in o] for o in output_records]
        for i in range(batch_size):
            info('------------------------------- in step {} {}th code ------------------------------------'.format(step_i, i))
            inp_code = id_to_code_fn(batch_data['input_seq'][i][1:-1])
            info('input  data: {}'.format(inp_code))
            for j in range(step_times):
                info('------------------------------- in step {} {}th code {} iter ------------------------------------'.format(step_i, i, j))
                p1 = output_records_list[j][0][i]
                p2 = output_records_list[j][1][i]
                sample_output_ids = output_records_list[j][5][i]
                try:
                    end_pos = sample_output_ids.index(inner_end_id)
                    end_pos += 1
                except ValueError as e:
                    end_pos = len(sample_output_ids)

                is_copy = output_records_list[j][2][i][:end_pos]
                copy_ids = output_records_list[j][3][i][:end_pos]
                copy_words = id_to_word_fn(copy_ids)
                copy_words = [w if c == 1 else '<SAMPLE>' for c, w in zip(is_copy, copy_words)]
                sample_output = output_records_list[j][4][i][:end_pos]
                sample_words = id_to_word_fn(sample_output)
                sample_words = ['<COPY>' if c== 1 else s for c, s in zip(is_copy, sample_words)]
                sample_output_ids = output_records_list[j][5][i][:end_pos]

                out = final_output[j][i]
                res = compile_result_list[j][i]
                info('compile result: {}'.format(res))
                out_code = id_to_code_fn(out)
                info('output data: {}'.format(out_code))
                info('position: {}, {}'.format(p1, p2))
                is_copy_str = [str(c) for c in is_copy]
                info('is_copy: {}'.format(' '.join(is_copy_str)))
                info('copy_output: {}'.format(' '.join(copy_words)))
                info('sample output: {}'.format(' '.join(sample_words)))
                info('part output: {}'.format(id_to_code_fn(sample_output_ids)))

    return multi_step_print_output_records

