import more_itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.problem_util import to_cuda
from common.util import PaddedList
from model.pretrain_model.transformer.transformer import Encoder, Decoder, Transformer


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pad_idx, n_head=2, n_layers=1, dropout=0.2, max_length=500,
                 bidirectional_decoder=False, model_type='seq2seq'):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_head = n_head
        self.dropout = dropout
        self.bidirectional_decoder = bidirectional_decoder
        self.model_type = model_type

        self.d_k_v_q = int(hidden_size/n_head)

        if model_type == 'seq2seq':
            self.context = Transformer(n_src_vocab=vocab_size, n_trg_vocab=vocab_size, src_pad_idx=pad_idx,
                                           trg_pad_idx=pad_idx, d_word_vec=hidden_size, d_model=hidden_size,
                                           d_inner=hidden_size, n_layers=n_layers, n_head=n_head, d_k=self.d_k_v_q,
                                           d_v=self.d_k_v_q, dropout=dropout, n_position=max_length,
                                           trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=True,
                                           bidirectional_decoder=bidirectional_decoder)

        elif model_type == 'only_encoder':
            self.context = nn.ModuleList([
                Encoder(n_src_vocab=vocab_size, n_position=max_length, d_word_vec=hidden_size, d_model=hidden_size,
                        d_inner=hidden_size, n_layers=n_layers, n_head=n_head, d_k=self.d_k_v_q, d_v=self.d_k_v_q,
                        pad_idx=pad_idx, dropout=dropout),
                nn.Linear(hidden_size, vocab_size)])

    def forward(self, inp_seq, inp_seq_len, inp_mask, target_seq, target_len, target_mask, masked_positions):
        if self.model_type == 'seq2seq':
            o = self.context(inp_seq, inp_seq)
        elif self.model_type == 'only_encoder':
            src_mask = get_pad_mask(inp_seq, self.pad_idx)
            o, *_ = self.context[0](inp_seq, src_mask)
            o = self.context[1](o)
        return [o]


def create_loss_fn(ignore_id):
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)

    def loss_fn(output_logit, target_seq):
        loss = cross_loss(output_logit.view(-1, output_logit.size(2)), target_seq.view(-1))
        return loss
    return loss_fn


def create_parse_input_batch_data_fn():
    def parse_input_tensor(batch_data, do_sample=False):
        input_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['input_seq'])))
        inp_seq_len = to_cuda(torch.LongTensor(batch_data['input_seq_len']))
        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_seq'])))
        target_seq_len = to_cuda(torch.LongTensor(batch_data['target_seq_len']))
        return input_seq, inp_seq_len, None, target_seq, target_seq_len, None, batch_data['masked_positions']
    return parse_input_tensor


def create_parse_target_batch_data_fn(ignore_id):
    def parse_target_tensor(batch_data):
        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_seq'], fill_value=ignore_id)))
        return [target_seq]
    return parse_target_tensor


def create_output_ids_fn():
    def create_output(model_output, model_input, do_sample):
        input_seq = model_input[0]
        output_seq = model_output[0]
        output_seq = torch.squeeze(torch.topk(F.softmax(output_seq, dim=-1), dim=-1, k=1)[1], dim=-1)

        masked_positions = model_input[6]
        positions = [[(i, p) for p in positions] for i, positions in enumerate(masked_positions)]
        positions = torch.LongTensor(list(more_itertools.flatten(positions))).to(input_seq.device)

        mask = torch.ones(input_seq.size(), device=input_seq.device).byte()
        mask[positions[:,0],positions[:,1]] = 0

        result = torch.where(mask, input_seq, output_seq)
        return result
    return create_output


