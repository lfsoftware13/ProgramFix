import os

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from common import torch_util
from common.args_util import to_cuda, get_model
from common.logger import init_a_file_logger, info
from common.opt import OpenAIAdam
from common.torch_util import create_sequence_length_mask, permute_last_dim_to_second, expand_tensor_sequence_len
from common.util import data_loader, show_process_map, PaddedList
from experiment.experiment_util import load_common_error_data, load_common_error_data_sample_100
from model.base_attention import TransformEncoderModel, TransformDecoderModel, TrueMaskedMultiHeaderAttention, \
    register_hook, PositionWiseFeedForwardNet, is_nan, record_is_nan
from read_data.load_data_vocabulary import create_common_error_vocabulary
from seq2seq.models import EncoderRNN, DecoderRNN
from vocabulary.word_vocabulary import Vocabulary


MAX_LENGTH = 500
IGNORE_TOKEN = -1
is_debug = False

class CCodeErrorDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transform=None):
        self.data_df = data_df[data_df['error_code_word_id'].map(lambda x: x is not None)]
        # print(self.data_df['error_code_word_id'])
        self.data_df = self.data_df[self.data_df['error_code_word_id'].map(lambda x: len(x) < MAX_LENGTH)]
        self.set_type = set_type
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)
        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]

        sample = {"error_tokens": self.data_df.iloc[index]['error_code_word_id'],
                  'error_length': len(self.data_df.iloc[index]['error_code_word_id']),
                  'includes': self.data_df.iloc[index]['includes']}
        if self.set_type != 'valid' or self.set_type != 'test':
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['ac_tokens'] = [begin_id] + self.data_df.iloc[index]['ac_code_word_id'] + [end_id]
            sample['ac_length'] = len(sample['ac_tokens'])
            sample['token_map'] = self.data_df.iloc[index]['token_map']
            sample['pointer_map'] = create_pointer_map(sample['ac_length'], sample['token_map'])
            sample['error_mask'] = self.data_df.iloc[index]['error_mask']
            sample['is_copy'] = [0] + self.data_df.iloc[index]['is_copy'] + [0]
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            sample['token_map'] = None
            sample['pointer_map'] = None
            sample['error_mask'] = None
            sample['is_copy'] = None
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


def create_pointer_map(ac_length, token_map):
    """
    the ac length includes begin and end label.
    :param ac_length:
    :param token_map:
    :return: map ac id to error pointer position with begin and end
    """
    pointer_map = [-1 for i in range(ac_length)]
    for error_i, ac_i in enumerate(token_map):
        if ac_i >= 0:
            pointer_map[ac_i + 1] = error_i
    last_point = -1
    for i in range(len(pointer_map)):
        if pointer_map[i] == -1:
            pointer_map[i] = last_point + 1
        else:
            last_point = pointer_map[i]
            if last_point + 1 >= len(token_map):
                last_point = len(token_map) - 2
    return pointer_map


class SinCosPositionEmbeddingModel(nn.Module):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4):
        super(SinCosPositionEmbeddingModel, self).__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, x, position_start_list=None):
        """

        :param x: has more than 3 dims
        :param position_start_list: len(position_start_list) == len(x.shape) - 2. default: [0] * num_dims.
                create position from start to start+length-1 for each dim.
        :param min_timescale:
        :param max_timescale:
        :return:
        """
        x_shape = list(x.shape)
        num_dims = len(x_shape) - 2
        channels = x_shape[-1]
        num_timescales = channels // (num_dims * 2)
        log_timescales_increment = (math.log(float(self.max_timescale) / float(self.min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = self.min_timescale * to_cuda(torch.exp(torch.range(0, num_timescales - 1) * -log_timescales_increment))
        # add moved position start index
        if position_start_list is None:
            position_start_list = [0] * num_dims
        for dim in range(num_dims):
            length = x_shape[dim + 1]
            # position = transform_to_cuda(torch.range(0, length-1))
            # create position from start to start+length-1 for each dim
            position = to_cuda(torch.range(position_start_list[dim], position_start_list[dim] + length - 1))
            scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad, 0, 0))
            for _ in range(dim + 1):
                signal = torch.unsqueeze(signal, dim=0)
            for _ in range(num_dims - dim - 1):
                signal = torch.unsqueeze(signal, dim=-2)
            x += signal
        return x


class IndexPositionEmbedding(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, max_len):
        super(IndexPositionEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)

    def forward(self, inputs, input_mask=None, start_index=0):
        batch_size = inputs.shape[0]
        input_sequence_len = inputs.shape[1]
        embedded_input = self.embedding(inputs)
        position_input_index = to_cuda(
            torch.unsqueeze(torch.arange(start_index, start_index + input_sequence_len), dim=0).expand(batch_size, -1)).long()
        if input_mask is not None:
            position_input_index = position_input_index.masked_fill_(~input_mask, MAX_LENGTH)
        position_input_embedded = self.position_embedding(position_input_index)
        position_input = torch.cat([position_input_embedded, embedded_input], dim=-1)
        return position_input

    def forward_position(self, inputs, input_mask=None):
        batch_size = inputs.shape[0]
        input_sequence_len = inputs.shape[1]
        position_input_index = to_cuda(
            torch.unsqueeze(torch.arange(0, input_sequence_len), dim=0).expand(batch_size, -1)).long()
        if input_mask is not None:
            position_input_index = position_input_index.masked_fill_(~input_mask, MAX_LENGTH)
        position_input_embedded = self.position_embedding(position_input_index)
        return position_input_embedded


class SelfAttentionPointerNetworkModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, encoder_stack_num, decoder_stack_num, start_label, end_label, dropout_p=0.1, num_heads=2, normalize_type=None, MAX_LENGTH=500):
        super(SelfAttentionPointerNetworkModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.encoder_stack_num = encoder_stack_num
        self.decoder_stack_num = decoder_stack_num
        self.start_label = start_label
        self.end_label = end_label
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.MAX_LENGTH = MAX_LENGTH

        # self.encode_embedding = nn.Embedding(vocabulary_size, hidden_size//2)
        # self.encode_position_embedding = SinCosPositionEmbeddingModel()
        self.encode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)
        # self.decode_embedding = nn.Embedding(vocabulary_size, hidden_size//2)
        # self.decode_position_embedding = SinCosPositionEmbeddingModel()
        self.decode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)

        self.encoder = TransformEncoderModel(hidden_size=self.hidden_size, encoder_stack_num=self.encoder_stack_num, dropout_p=self.dropout_p, num_heads=self.num_heads, normalize_type=self.normalize_type)
        self.decoder = TransformDecoderModel(hidden_size=hidden_size, decoder_stack_num=self.decoder_stack_num, dropout_p=self.dropout_p, num_heads=self.num_heads, normalize_type=self.normalize_type)

        # self.copy_linear = nn.Linear(self.hidden_size, 1)
        self.copy_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=1, hidden_layer_count=1)

        # self.position_pointer = TrueMaskedMultiHeaderAttention(hidden_size=self.hidden_size, num_heads=1, attention_type='scaled_dot_product')
        # self.position_pointer = nn.Linear(self.hidden_size, self.hidden_size)
        self.position_pointer = PositionWiseFeedForwardNet(self.hidden_size, self.hidden_size, self.hidden_size//2, 1)
        # self.output_linear = nn.Linear(self.hidden_size, self.vocabulary_size)
        self.output_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=vocabulary_size, hidden_layer_count=1)

    def create_next_output(self, copy_output, value_output, pointer_output, input):
        """

        :param copy_output: [batch, sequence, dim**]
        :param value_output: [batch, sequence, dim**, vocabulary_size]
        :param pointer_output: [batch, sequence, dim**, encode_length]
        :param input: [batch, encode_length, dim**]
        :return: [batch, sequence, dim**]
        """
        is_copy = (copy_output > 0.5)
        _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
        _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
        point_id = torch.gather(input, dim=-1, index=pointer_pos_in_input.squeeze(dim=-1))
        next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
        return next_output

    def decode_step(self, decode_input, decode_mask, encode_value, encode_mask, position_embedded):
        # record_is_nan(encode_value, 'encode_value')
        decoder_output = self.decoder(decode_input, decode_mask, encode_value, encode_mask)
        # record_is_nan(decoder_output, 'decoder_output')

        is_copy = F.sigmoid(self.copy_linear(decoder_output).squeeze(dim=-1))
        # record_is_nan(is_copy, 'is_copy in model: ')
        pointer_ff = self.position_pointer(decoder_output)
        # record_is_nan(pointer_ff, 'pointer_ff in model: ')
        pointer_output = torch.bmm(pointer_ff, torch.transpose(position_embedded, dim0=-1, dim1=-2))
        # record_is_nan(pointer_output, 'pointer_output in model: ')
        if encode_mask is not None:
            dim_len = len(pointer_output.shape)
            pointer_output.masked_fill_(~encode_mask.view(encode_mask.shape[0], *[1 for i in range(dim_len-2)], encode_mask.shape[-1]), -float('inf'))
            # pointer_output = torch.where(torch.unsqueeze(encode_mask, dim=1), pointer_output, to_cuda(torch.Tensor([float('-inf')])))

        value_output = self.output_linear(decoder_output)
        return is_copy, value_output, pointer_output

    def forward(self, input, input_mask, output, output_mask):
        position_input = self.encode_position_embedding(input, input_mask)
        position_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
        encode_value = self.encoder(position_input, input_mask)

        position_output = self.decode_position_embedding(output, output_mask)
        is_copy, value_output, pointer_output = self.decode_step(position_output, output_mask, encode_value, input_mask,
                                                                 position_embedded=position_input_embedded)
        return is_copy, value_output, pointer_output

class RNNPointerNetworkModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_layers, start_label, end_label, dropout_p=0.1, MAX_LENGTH=500):
        super(RNNPointerNetworkModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.start_label = start_label
        self.end_label = end_label
        self.dropout_p = dropout_p
        self.MAX_LENGTH = MAX_LENGTH

        self.encode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)
        self.decode_position_embedding = IndexPositionEmbedding(vocabulary_size, self.hidden_size//2, MAX_LENGTH+1)

        self.encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, input_size=self.hidden_size, hidden_size=self.hidden_size//2,
                                  n_layers=num_layers, bidirectional=True, rnn_cell='lstm',
                                  input_dropout_p=self.dropout_p, dropout_p=self.dropout_p, variable_lengths=False,
                                  embedding=None, update_embedding=True)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, hidden_size=hidden_size,
                                  sos_id=start_label, eos_id=end_label, n_layers=num_layers, rnn_cell='lstm',
                                  bidirectional=False, input_dropout_p=self.dropout_p, dropout_p=self.dropout_p,
                                  use_attention=True)

        self.copy_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=1, hidden_layer_count=1)

        # self.position_pointer = TrueMaskedMultiHeaderAttention(hidden_size=self.hidden_size, num_heads=1, attention_type='scaled_dot_product')
        # self.position_pointer = nn.Linear(self.hidden_size, self.hidden_size)
        self.position_pointer = PositionWiseFeedForwardNet(self.hidden_size, self.hidden_size, self.hidden_size//2, 1)
        # self.output_linear = nn.Linear(self.hidden_size, self.vocabulary_size)
        self.output_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                      output_size=vocabulary_size, hidden_layer_count=1)

    def create_next_output(self, copy_output, value_output, pointer_output, input):
        """

        :param copy_output: [batch, sequence, dim**]
        :param value_output: [batch, sequence, dim**, vocabulary_size]
        :param pointer_output: [batch, sequence, dim**, encode_length]
        :param input: [batch, encode_length, dim**]
        :return: [batch, sequence, dim**]
        """
        is_copy = (copy_output > 0.5)
        _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
        _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
        point_id = torch.gather(input, dim=-1, index=pointer_pos_in_input.squeeze(dim=-1))
        next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
        return next_output

    def decode_step(self, decode_input, encoder_hidden, decode_mask, encode_value, encode_mask, position_embedded, teacher_forcing_ratio=1):
        # record_is_nan(encode_value, 'encode_value')
        decoder_output, hidden, _ = self.decoder(inputs=decode_input, encoder_hidden=encoder_hidden, encoder_outputs=encode_value,
                                            encoder_mask=~encode_mask, teacher_forcing_ratio=teacher_forcing_ratio)
        decoder_output = torch.stack(decoder_output, dim=1)
        # record_is_nan(decoder_output, 'decoder_output')
        if decode_mask is not None:
            decode_mask = decode_mask[:, 1:] if teacher_forcing_ratio == 1 else decode_mask
            decoder_output = decoder_output * decode_mask.view(decode_mask.shape[0], decode_mask.shape[1], *[1 for i in range(len(decoder_output.shape)-2)]).float()

        is_copy = F.sigmoid(self.copy_linear(decoder_output).squeeze(dim=-1))
        # record_is_nan(is_copy, 'is_copy in model: ')
        pointer_ff = self.position_pointer(decoder_output)
        # record_is_nan(pointer_ff, 'pointer_ff in model: ')
        pointer_output = torch.bmm(pointer_ff, torch.transpose(position_embedded, dim0=-1, dim1=-2))
        # record_is_nan(pointer_output, 'pointer_output in model: ')
        if encode_mask is not None:
            dim_len = len(pointer_output.shape)
            pointer_output.masked_fill_(~encode_mask.view(encode_mask.shape[0], *[1 for i in range(dim_len-2)], encode_mask.shape[-1]), -float('inf'))
            # pointer_output = torch.where(torch.unsqueeze(encode_mask, dim=1), pointer_output, to_cuda(torch.Tensor([float('-inf')])))

        value_output = self.output_linear(decoder_output)
        return is_copy, value_output, pointer_output, hidden

    def forward(self, inputs, input_mask, output, output_mask, test=False):
        if test:
            return self.forward_test(inputs, input_mask)
        position_input = self.encode_position_embedding(inputs, input_mask)
        position_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
        encode_value, hidden = self.encoder(position_input)
        encoder_hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        position_output = self.decode_position_embedding(inputs=output, input_mask=output_mask)
        is_copy, value_output, pointer_output, _ = self.decode_step(decode_input=position_output, encoder_hidden=encoder_hidden,
                                                                 decode_mask=output_mask, encode_value=encode_value,
                                                                 position_embedded=position_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=1)
        return is_copy, value_output, pointer_output

    def forward_test(self, input, input_mask):
        position_input = self.encode_position_embedding(input, input_mask)
        position_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
        encode_value, hidden = self.encoder(position_input)
        hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        batch_size = list(input.shape)[0]
        continue_mask = to_cuda(torch.Tensor([1 for i in range(batch_size)])).byte()
        outputs = to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)]))
        is_copy_stack = []
        value_output_stack = []
        pointer_stack = []
        output_stack = []

        for i in range(self.MAX_LENGTH):
            cur_input_mask = continue_mask.view(continue_mask.shape[0], 1)
            output_embed = self.decode_position_embedding(inputs=outputs, input_mask=cur_input_mask, start_index=i)

            is_copy, value_output, pointer_output, hidden = self.decode_step(decode_input=output_embed, encoder_hidden=hidden,
                                                                 decode_mask=cur_input_mask, encode_value=encode_value,
                                                                 position_embedded=position_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=0)
            is_copy_stack.append(is_copy)
            value_output_stack.append(value_output)
            pointer_stack.append(pointer_output)

            outputs = self.create_next_output(is_copy, value_output, pointer_output, input)
            output_stack.append(outputs)

            cur_continue = torch.ne(outputs, self.end_label).view(outputs.shape[0])
            continue_mask = continue_mask & cur_continue

            if torch.sum(continue_mask) == 0:
                break

        is_copy_result = torch.cat(is_copy_stack, dim=1)
        value_result = torch.cat(value_output_stack, dim=1)
        pointer_result = torch.cat(pointer_stack, dim=1)
        output_result = torch.cat(output_stack, dim=1)
        return output_result, is_copy_result, value_result, pointer_result


def parse_input_batch_data(batch_data):
    error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
    max_input = max(batch_data['error_length'])
    input_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['error_length'])), max_len=max_input))

    ac_tokens_decoder_input = [bat[:-1] for bat in batch_data['ac_tokens']]
    ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_input)))
    ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_input]
    max_output = max(ac_tokens_length)
    output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

    # info('error_tokens shape: {}, input_mask shape: {}, ac_tokens shape: {}, output_mask shape: {}'.format(
    #     error_tokens.shape, input_mask.shape, ac_tokens.shape, output_mask.shape))
    # info('error_length: {}'.format(batch_data['error_length']))
    # info('ac_length: {}'.format(batch_data['ac_length']))

    return error_tokens, input_mask, ac_tokens, output_mask


def parse_rnn_input_batch_data(batch_data):
    error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
    max_input = max(batch_data['error_length'])
    input_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['error_length'])), max_len=max_input))

    # ac_tokens_decoder_input = [bat[:-1] for bat in batch_data['ac_tokens']]
    ac_tokens_decoder_input = batch_data['ac_tokens']
    ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_input)))
    ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_input]
    max_output = max(ac_tokens_length)
    output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

    # info('error_tokens shape: {}, input_mask shape: {}, ac_tokens shape: {}, output_mask shape: {}'.format(
    #     error_tokens.shape, input_mask.shape, ac_tokens.shape, output_mask.shape))
    # info('error_length: {}'.format(batch_data['error_length']))
    # info('ac_length: {}'.format(batch_data['ac_length']))

    return error_tokens, input_mask, ac_tokens, output_mask


def parse_target_batch_data(batch_data):
    ac_tokens_decoder_output = [bat[1:] for bat in batch_data['ac_tokens']]
    target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_output, fill_value=IGNORE_TOKEN)))

    ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_output]
    max_output = max(ac_tokens_length)
    output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

    pointer_map = [bat[1:] for bat in batch_data['pointer_map']]
    target_pointer_output = to_cuda(torch.LongTensor(PaddedList(pointer_map, fill_value=IGNORE_TOKEN)))

    is_copy = [bat[1:] for bat in batch_data['is_copy']]
    target_is_copy = to_cuda(torch.Tensor(PaddedList(is_copy)))

    return target_is_copy, target_pointer_output, target_ac_tokens, output_mask

def create_combine_loss_fn(copy_weight=1, pointer_weight=1, value_weight=1, average_value=True):
    copy_loss_fn = nn.BCELoss(reduce=False)
    pointer_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)
    value_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)

    def combine_total_loss(is_copy, pointer_log_probs, value_log_probs, target_copy, target_pointer, target_value, output_mask):
        # record_is_nan(is_copy, 'is_copy: ')
        # record_is_nan(pointer_log_probs, 'pointer_log_probs: ')
        # record_is_nan(value_log_probs, 'value_log_probs: ')
        output_mask_float = output_mask.float()
        # info('output_mask_float: {}, mask batch: {}'.format(str(output_mask_float), torch.sum(output_mask_float, dim=-1)))
        # info('target_copy: {}, mask batch: {}'.format(str(target_copy), torch.sum(target_copy, dim=-1)))

        copy_loss = copy_loss_fn(is_copy, target_copy)
        # record_is_nan(copy_loss, 'copy_loss: ')
        copy_loss = copy_loss * output_mask_float
        # record_is_nan(copy_loss, 'copy_loss with mask: ')

        pointer_log_probs = permute_last_dim_to_second(pointer_log_probs)
        pointer_loss = pointer_loss_fn(pointer_log_probs, target_pointer)
        # record_is_nan(pointer_loss, 'pointer_loss: ')
        pointer_loss = pointer_loss * output_mask_float
        # record_is_nan(pointer_loss, 'pointer_loss with mask: ')

        value_log_probs = permute_last_dim_to_second(value_log_probs)
        value_loss = value_loss_fn(value_log_probs, target_value)
        # record_is_nan(value_loss, 'value_loss: ')
        value_mask_float = output_mask_float * (~target_copy.byte()).float()
        # info('value_mask_float: {}, mask batch: {}'.format(str(value_mask_float), torch.sum(value_mask_float, dim=-1)))
        value_loss = value_loss * value_mask_float
        # record_is_nan(value_loss, 'value_loss with mask: ')
        total_count = torch.sum(output_mask_float)
        if average_value:
            value_count = torch.sum(value_mask_float)
            value_loss = value_loss / value_count * total_count

        total_loss = copy_weight*copy_loss + pointer_weight*pointer_loss + value_weight*value_loss
        # total_loss = pointer_weight*pointer_loss
        return torch.sum(total_loss) / total_count
    return combine_total_loss


def create_output_ids(is_copy, value_output, pointer_output, error_tokens):
    """

    :param copy_output: [batch, sequence, dim**]
    :param value_output: [batch, sequence, dim**, vocabulary_size]
    :param pointer_output: [batch, sequence, dim**, encode_length]
    :param input: [batch, encode_length, dim**]
    :return: [batch, sequence, dim**]
    """
    is_copy = (is_copy > 0.5)
    _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
    _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
    pointer_pos_in_input = pointer_pos_in_input.squeeze(dim=-1)
    point_id = torch.gather(error_tokens, dim=-1, index=pointer_pos_in_input)
    next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
    return next_output


def train(model, dataset, batch_size, loss_function, optimizer, clip_norm, epoch_ratio):
    total_loss = to_cuda(torch.Tensor([0]))
    count = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    model.train()

    with tqdm(total=(len(dataset)*epoch_ratio)) as pbar:
        for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True, epoch_ratio=epoch_ratio):
            model.zero_grad()
            # target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
            # target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'], fill_value=IGNORE_TOKEN)))
            # target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
            # max_output = max(batch_data['ac_length'])

            # model_input = parse_input_batch_data(batch_data)
            model_input = parse_rnn_input_batch_data(batch_data)
            is_copy, value_output, pointer_output = model.forward(*model_input)
            if steps % 100 == 0:
                pointer_output_id = torch.squeeze(torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                info(pointer_output_id)
                # print(pointer_output_id)

            model_target = parse_target_batch_data(batch_data)
            target_is_copy, target_pointer_output, target_ac_tokens, output_mask = model_target
            loss = loss_function(is_copy, pointer_output, value_output, *model_target)
            # record_is_nan(loss, 'total loss in train:')


            # if steps == 0:
            #     pointer_probs = F.softmax(pointer_output, dim=-1)
                # print('pointer output softmax: ', pointer_probs)
                # print('pointer output: ', torch.squeeze(torch.topk(pointer_probs, k=1, dim=-1)[1], dim=-1))
                # print('target_pointer_output: ', target_pointer_output)

            batch_count = torch.sum(output_mask).float()
            batch_loss = loss * batch_count
            # loss = batch_loss / batch_count

            # if clip_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            loss.backward()
            optimizer.step()

            output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
            batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
            accuracy = batch_accuracy / batch_count
            total_accuracy += batch_accuracy
            total_loss += batch_loss

            step_output = 'in train step {}  loss: {}, accracy: {}'.format(steps, loss.data.item(), accuracy.data.item())
            # print(step_output)
            info(step_output)

            count += batch_count
            steps += 1
            pbar.update(batch_size)

    return (total_loss/count).item(), (total_accuracy/count).item()


def evaluate(model, dataset, batch_size, loss_function, do_sample=False):
    total_loss = to_cuda(torch.Tensor([0]))
    count = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    model.eval()

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True):
                model.zero_grad()
                # target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
                # target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'])))
                # target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
                # max_output = max(batch_data['ac_length'])
                # output_mask = create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['ac_length'])),
                #                                           max_len=max_output)


                # model_input = parse_input_batch_data(batch_data)
                model_input = parse_rnn_input_batch_data(batch_data)
                model_target = parse_target_batch_data(batch_data)
                target_is_copy, target_pointer_output, target_ac_tokens, output_mask = model_target
                # is_copy, value_output, pointer_output = model.forward(*model_input)
                if do_sample:
                    _, is_copy, value_output, pointer_output = model.forward(*model_input[:2], None, None, test=True)
                    predict_len = is_copy.shape[1]
                    target_len = target_is_copy.shape[1]
                    expand_len = max(predict_len, target_len)
                    is_copy = expand_tensor_sequence_len(is_copy, max_len=expand_len, fill_value=0)
                    value_output = expand_tensor_sequence_len(value_output, max_len=expand_len, fill_value=0)
                    pointer_output = expand_tensor_sequence_len(pointer_output, max_len=expand_len, fill_value=0)
                    target_is_copy = expand_tensor_sequence_len(target_is_copy, max_len=expand_len, fill_value=0)
                    target_pointer_output = expand_tensor_sequence_len(target_pointer_output, max_len=expand_len, fill_value=IGNORE_TOKEN)
                    target_ac_tokens = expand_tensor_sequence_len(target_ac_tokens, max_len=expand_len, fill_value=IGNORE_TOKEN)
                    output_mask = expand_tensor_sequence_len(output_mask, max_len=expand_len, fill_value=0)
                else:
                    is_copy, value_output, pointer_output = model.forward(*model_input)


                loss = loss_function(is_copy, pointer_output, value_output, target_is_copy, target_pointer_output,
                                     target_ac_tokens, output_mask)
                # if steps == 0:
                #     pointer_probs = F.softmax(pointer_output, dim=-1)
                    # print('pointer output softmax: ', pointer_probs)
                    # print('pointer output: ', torch.squeeze(torch.topk(pointer_probs, k=1, dim=-1)[1], dim=-1))
                    # print('target_pointer_output: ', target_pointer_output)

                batch_count = torch.sum(output_mask).float()
                batch_loss = loss * batch_count
                # loss = batch_loss / batch_count

                output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
                batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
                accuracy = batch_accuracy / batch_count
                total_accuracy += batch_accuracy
                total_loss += batch_loss

                step_output = 'in evaluate step {}  loss: {}, accracy: {}'.format(steps, loss.data.item(), accuracy.data.item())
                # print(step_output)
                info(step_output)

                count += batch_count
                steps += 1
                pbar.update(batch_size)

        # output_result, is_copy, value_output, pointer_output = model.forward_test(*parse_test_batch_data(batch_data))

    return (total_loss / count).item(), (total_accuracy / count).item()


def train_and_evaluate(batch_size, hidden_size, num_heads, encoder_stack_num, decoder_stack_num, num_layers, dropout_p, learning_rate, epoches, saved_name, load_name=None, epoch_ratio=1.0, normalize_type='layer', clip_norm=80, parallel=False, logger_file_path=None):
    valid_loss = 0
    test_loss = 0
    valid_accuracy = 0
    test_accuracy = 0
    sample_valid_loss = 0
    sample_test_loss = 0
    sample_valid_accuracy = 0
    sample_test_accuracy = 0

    save_path = os.path.join(config.save_model_root, saved_name)
    if load_name is not None:
        load_path = os.path.join(config.save_model_root, load_name)

    if logger_file_path is not None:
        init_a_file_logger(logger_file_path)

    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token, addition_tokens=addition_tokens)

    begin_tokens_id = [vocabulary.word_to_id(i) for i in begin_tokens]
    end_tokens_id = [vocabulary.word_to_id(i) for i in end_tokens]
    unk_token_id = vocabulary.word_to_id(unk_token)
    addition_tokens_id = [vocabulary.word_to_id(i) for i in addition_tokens]

    if is_debug:
        data_dict = load_common_error_data_sample_100()
    else:
        data_dict = load_common_error_data()
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name) for dd, name in
                zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets

    # model = SelfAttentionPointerNetworkModel(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
    #                                  encoder_stack_num=encoder_stack_num, decoder_stack_num=decoder_stack_num,
    #                                  start_label=begin_tokens_id[0], end_label=end_tokens_id[0], dropout_p=dropout_p,
    #                                  num_heads=num_heads, normalize_type=normalize_type, MAX_LENGTH=MAX_LENGTH)
    model = RNNPointerNetworkModel(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
                                     num_layers=num_layers, start_label=begin_tokens_id[0], end_label=end_tokens_id[0],
                                   dropout_p=dropout_p, MAX_LENGTH=MAX_LENGTH)

    model = get_model(model)

    if load_name is not None:
        torch_util.load_model(model, load_path, map_location={'cuda:1': 'cuda:0'})

    loss_fn = create_combine_loss_fn(average_value=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = OpenAIAdam(model.parameters(), lr=learning_rate, schedule='warmup_linear', warmup=0.002, t_total=epoches * len(train_dataset)//batch_size, max_grad_norm=clip_norm)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if load_name is not None:
        valid_loss, valid_accuracy = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                              loss_function=loss_fn)
        test_loss, test_accuracy = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                            loss_function=loss_fn)
        sample_valid_loss, sample_valid_accuracy = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                              loss_function=loss_fn, do_sample=True)
        sample_test_loss, sample_test_accuracy = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                            loss_function=loss_fn, do_sample=True)
        evaluate_output = 'evaluate: valid loss of {}, test loss of {}, ' \
                          'valid_accuracy result of {}, test_accuracy result of {}' \
                          'sample valid loss: {}, sample test loss: {}, ' \
                          'sample valid accuracy: {}, sample test accuracy: {}'.format(
            valid_loss, test_loss, valid_accuracy, test_accuracy,
            sample_valid_loss, sample_test_loss, sample_valid_accuracy, sample_test_accuracy)
        print(evaluate_output)
        info(evaluate_output)
        pass

    for epoch in range(epoches):
        train_loss, train_accuracy = train(model=model, dataset=train_dataset, batch_size=batch_size,
                                           loss_function=loss_fn, optimizer=optimizer, clip_norm=clip_norm,
                                           epoch_ratio=epoch_ratio)
        valid_loss, valid_accuracy = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                              loss_function=loss_fn)
        test_loss, test_accuracy = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                            loss_function=loss_fn)
        sample_valid_loss, sample_valid_accuracy = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                              loss_function=loss_fn, do_sample=True)
        sample_test_loss, sample_test_accuracy = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                            loss_function=loss_fn, do_sample=True)
        # scheduler.step(train_loss)
        # print('epoch {}: train loss: {}, accuracy: {}'.format(epoch, train_loss, train_accuracy))
        epoch_output = 'epoch {}: train loss of {}, valid loss of {}, test loss of {}, ' \
                       'train_accuracy result of {} valid_accuracy result of {}, test_accuracy result of {}' \
                       'sample valid loss: {}, sample test loss: {}, ' \
                       'sample valid accuracy: {}, sample test accuracy: {}'.format(
            epoch, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy,
            sample_valid_loss, sample_test_loss, sample_valid_accuracy, sample_test_accuracy)
        print(epoch_output)
        info(epoch_output)

        if not is_debug:
            torch_util.save_model(model, save_path+str(epoch))


if __name__ == '__main__':
    # train_and_evaluate(batch_size=12, hidden_size=400, num_heads=3, encoder_stack_num=4, decoder_stack_num=4, num_layers=3, dropout_p=0,
    #                    learning_rate=6.25e-5, epoches=40, saved_name='SelfAttentionPointer.pkl', load_name='SelfAttentionPointer.pkl',
    #                    epoch_ratio=0.25, normalize_type=None, clip_norm=1, parallel=False, logger_file_path='log/SelfAttentionPointer.log')

    train_and_evaluate(batch_size=16, hidden_size=400, num_heads=0, encoder_stack_num=0, decoder_stack_num=0, num_layers=3, dropout_p=0,
                       learning_rate=6.25e-5, epoches=40, saved_name='RNNPointerAllLoss.pkl', load_name=None,
                       epoch_ratio=0.25, normalize_type=None, clip_norm=1, parallel=False, logger_file_path='log/RNNPointerAllLoss.log')







