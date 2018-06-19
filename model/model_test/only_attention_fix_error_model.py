import math
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

import config
from common import torch_util
from common.torch_util import spilt_heads, create_sequence_length_mask
from common.util import data_loader, show_process_map, PaddedList
from experiment.experiment_util import load_common_error_data, load_common_error_data_sample_100
from experiment.parse_xy_util import choose_token_random_batch, combine_spilt_tokens_batch
from read_data.load_data_vocabulary import create_common_error_vocabulary
from vocabulary.word_vocabulary import Vocabulary


is_cuda = True
gpu_index = 1
MAX_LENGTH = 500
TARGET_PAD_TOKEN = -1
is_debug = False


def transform_to_cuda(x):
    if is_cuda:
        x = x.cuda(gpu_index)
    return x


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
            sample['ac_tokens'] = self.data_df.iloc[index]['ac_code_word_id']
            sample['ac_length'] = len(self.data_df.iloc[index]['ac_code_word_id'])
            sample['token_map'] = self.data_df.iloc[index]['token_map']
            sample['error_mask'] = self.data_df.iloc[index]['error_mask']
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            sample['token_map'] = None
            sample['error_mask'] = None
        return sample



    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


def add_position_encode(x, position_start_list=None, min_timescale=1.0, max_timescale=1.0e4):
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
    log_timescales_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(transform_to_cuda(torch.range(0, num_timescales-1)) * -log_timescales_increment)
    # add moved position start index
    if position_start_list is None:
        position_start_list = [0] * num_dims
    for dim in range(num_dims):
        length = x_shape[dim + 1]
        # position = transform_to_cuda(torch.range(0, length-1))
        # create position from start to start+length-1 for each dim
        position = transform_to_cuda(torch.range(position_start_list[dim], position_start_list[dim] + length - 1))
        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = F.pad(signal, (prepad, postpad, 0, 0))
        for _ in range(dim + 1):
            signal = torch.unsqueeze(signal, dim=0)
        for _ in range(num_dims - dim -1):
            signal = torch.unsqueeze(signal, dim=-2)
        x += signal
    return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, value_hidden_size, output_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.output_dim = output_dim
        self.value_hidden_size = value_hidden_size
        self.transform_output = transform_to_cuda(nn.Linear(value_hidden_size, output_dim))

    def forward(self, query, key, value, value_mask=None):
        """
        query = [batch, -1, query_seq, query_hidden_size]
        key = [batch, -1, value_seq, query_hidden_size]
        value = [batch, -1, value_seq, value_hidden_size]
        the len(shape) of query, key, value must be the same
        Create attention value of query_sequence according to value_sequence(memory).
        :param query:
        :param key:
        :param value:
        :param value_mask: [batch, max_len]. consist of 0 and 1.
        :return: attention_value = [batch, -1, query_seq, output_hidden_size]
        """
        query_shape = list(query.shape)
        key_shape = list(key.shape)
        value_shape = list(value.shape)
        scaled_value = math.sqrt(key_shape[-1])

        # [batch, -1, query_hidden_size, value_seq] <- [batch, -1, value_seq, query_hidden_size]
        key = torch.transpose(key, dim0=-2, dim1=-1)

        # [batch, -1, query_seq, value_seq] = [batch, -1, query_seq, query_hidden_size] * [batch, -1, query_hidden_size, value_seq]
        query_3d = query.contiguous().view(-1, query_shape[-2], query_shape[-1])
        key_3d = key.contiguous().view(-1, key_shape[-1], key_shape[-2])
        qk_dotproduct = torch.bmm(query_3d, key_3d).view(*query_shape[:-2], query_shape[-2], key_shape[-2])
        scaled_qk_dotproduct = qk_dotproduct/scaled_value

        # mask the padded token value
        if value_mask is not None:
            dim_len = len(list(scaled_qk_dotproduct.shape))
            scaled_qk_dotproduct.data.masked_fill_(~value_mask.view(value_shape[0], *[1 for i in range(dim_len-2)], value_shape[-2]), -float('inf'))

        weight_distribute = F.softmax(scaled_qk_dotproduct, dim=-1)
        weight_shape = list(weight_distribute.shape)
        attention_value = torch.bmm(weight_distribute.view(-1, *weight_shape[-2:]), value.contiguous().view(-1, *value_shape[-2:]))
        attention_value = attention_value.view(*weight_shape[:-2], *list(attention_value.shape)[-2:])
        transformed_output = self.transform_output(attention_value)
        return transformed_output


class MaskedMultiHeaderAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, attention_type='scaled_dot_product'):
        super(MaskedMultiHeaderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_hidden_size = int(hidden_size/num_heads)

        self.query_linear = transform_to_cuda(nn.Linear(hidden_size, hidden_size))
        self.key_linear = transform_to_cuda(nn.Linear(hidden_size, hidden_size))
        self.value_linear = transform_to_cuda(nn.Linear(hidden_size, hidden_size))
        if attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention(self.attention_hidden_size, self.attention_hidden_size)
        else:
            raise Exception('no such attention_type: {}'.format(attention_type))

        self.output_linear = transform_to_cuda(nn.Linear(hidden_size, hidden_size))

    def forward(self, inputs, memory, memory_mask=None):
        """
        query = [batch, sequence, hidden_size]
        memory = [batch, sequence, hidden_size]

        :param query:
        :param key:
        :param value:
        :return:
        """
        query = self.query_linear(inputs)
        key = self.key_linear(memory)
        value = self.value_linear(memory)

        query_shape = list(query.shape)

        split_query = spilt_heads(query, self.num_heads)
        split_key = spilt_heads(key, self.num_heads)
        split_value = spilt_heads(value, self.num_heads)

        atte_value = self.attention.forward(split_query, split_key, split_value, value_mask=memory_mask)
        atte_value = torch.transpose(atte_value, dim0=-3, dim1=-2).contiguous().view(query_shape[:-1] + [-1])

        output_value = self.output_linear(atte_value)
        return output_value


class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, hidden_layer_count=1):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param hidden_layer_count: must >= 1
        """
        super(PositionWiseFeedForwardNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        if hidden_layer_count <= 0:
            raise Exception('at least one hidden layer')
        self.ff = transform_to_cuda(nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        ))
        for i in range(hidden_layer_count-1):
            self.ff.add_module('hidden_' + str(i), transform_to_cuda(nn.Linear(hidden_size, hidden_size)))
            self.ff.add_module('relu_' + str(i), transform_to_cuda(nn.ReLU()))

        self.ff.add_module('output', transform_to_cuda(nn.Linear(hidden_size, output_size)))

    def forward(self, x):
        return self.ff(x)


class SelfAttentionEncoder(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(SelfAttentionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.dropout_p = dropout_p

        self.dropout = transform_to_cuda(nn.Dropout(dropout_p))

        self.self_attention = MaskedMultiHeaderAttention(hidden_size, num_heads)
        self.ff = PositionWiseFeedForwardNet(hidden_size, hidden_size, hidden_size, hidden_layer_count=1)
        if normalize_type == 'layer':
            self.self_attention_normalize = transform_to_cuda(nn.LayerNorm(normalized_shape=hidden_size))
            self.ff_normalize = transform_to_cuda(nn.LayerNorm(normalized_shape=hidden_size))

    def forward(self, input, input_mask):
        atte_value = self.dropout(self.self_attention.forward(input, input, memory_mask=input_mask))
        if self.normalize_type is not None:
            atte_value = self.self_attention_normalize(atte_value) + atte_value

        ff_value = self.dropout(self.ff.forward(atte_value))
        if self.normalize_type is not None:
            ff_value = self.ff_normalize(ff_value) + ff_value
        return ff_value


class SelfAttentionDecoder(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(SelfAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.dropout_p = dropout_p

        self.dropout = transform_to_cuda(nn.Dropout(dropout_p))

        self.input_self_attention = MaskedMultiHeaderAttention(hidden_size, num_heads)
        self.attention = MaskedMultiHeaderAttention(hidden_size, num_heads)
        self.ff = PositionWiseFeedForwardNet(hidden_size, hidden_size, hidden_size, hidden_layer_count=1)

        if normalize_type is not None:
            self.self_attention_normalize = transform_to_cuda(nn.LayerNorm(normalized_shape=hidden_size))
            self.attention_normalize = transform_to_cuda(nn.LayerNorm(normalized_shape=hidden_size))
            self.ff_normalize = transform_to_cuda(nn.LayerNorm(normalized_shape=hidden_size))

    def forward(self, input, input_mask, encoder_output, encoder_mask):
        self_atte_value = self.dropout(self.input_self_attention(input, input, input_mask))
        if self.normalize_type is not None:
            self_atte_value = self.self_attention_normalize(self_atte_value)

        atte_value = self.dropout(self.attention(self_atte_value, encoder_output, encoder_mask))
        if self.normalize_type is not None:
            atte_value = self.attention_normalize(atte_value)

        ff_value = self.dropout(self.ff(atte_value))
        if self.normalize_type is not None:
            ff_value = self.ff_normalize(ff_value)

        return ff_value


class OnlyAttentionFixErrorModelWithoutInputEmbedding(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, sequence_max_length, num_heads, start_label, end_label, pad_label, dropout_p=0.1, encoder_stack_num=1, decoder_stack_num=1, normalize_type='layer'):
        super(OnlyAttentionFixErrorModelWithoutInputEmbedding, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.sequence_max_length = sequence_max_length
        self.start_label = start_label
        self.end_label = end_label
        self.pad_label = pad_label
        self.dropout_p = dropout_p

        self.dropout = transform_to_cuda(nn.Dropout(dropout_p))

        self.output_embedding = transform_to_cuda(nn.Embedding(vocabulary_size, hidden_size))
        self.position_encode_fn = add_position_encode

        if encoder_stack_num <= 0 or decoder_stack_num <= 0:
            raise Exception('stack_num should greater than 0: {}, {}'.format(encoder_stack_num, decoder_stack_num))

        self.encoder_list = nn.ModuleList()
        for i in range(encoder_stack_num):
            self.encoder_list.add_module('encoder_{}' + str(i), SelfAttentionEncoder(hidden_size, dropout_p=dropout_p, num_heads=num_heads, normalize_type=normalize_type))

        self.decoder_list = nn.ModuleList()
        for i in range(decoder_stack_num):
            self.decoder_list.add_module('decoder_{}' + str(i), SelfAttentionDecoder(hidden_size, dropout_p=dropout_p, num_heads=num_heads, normalize_type=normalize_type))

        self.output = transform_to_cuda(nn.Linear(hidden_size, vocabulary_size))

    def do_input_position(self, inputs, input_lengths):
        # input embedding
        if self.position_encode_fn is not None:
            position_input = self.dropout(self.position_encode_fn(inputs))
        else:
            position_input = inputs
        input_mask = create_sequence_length_mask(input_lengths, torch.max(input_lengths).item(), gpu_index=gpu_index)
        return input_mask, position_input

    def do_encode(self, input_mask, position_input_embed):
        # encoder for n times
        encode_value = position_input_embed
        for encoder in self.encoder_list:
            encode_value = encoder(encode_value, input_mask)
        return encode_value

    def forward(self, inputs_embed, input_lengths, outputs, output_lengths):
        input_mask, position_input_embed = self.do_input_position(inputs_embed, input_lengths)

        encode_value = self.do_encode(input_mask, position_input_embed)

        # output embedding
        outputs_embed = self.dropout(self.output_embedding(outputs))
        position_output = self.dropout(self.position_encode_fn(outputs_embed))
        output_mask = create_sequence_length_mask(output_lengths, torch.max(output_lengths).item(), gpu_index=gpu_index)

        decoder_value = self.do_decode(position_output, output_mask, encode_value, input_mask)
        output_value = self.output(decoder_value)
        return output_value

    def do_decode(self, position_output_embed, output_mask, encode_value, input_mask):
        # decoder for n times
        decoder_value = position_output_embed
        for decoder in self.decoder_list:
            decoder_value = decoder(decoder_value, output_mask, encode_value, input_mask)
        return decoder_value

    def forward_test(self, inputs_embed, input_lengths, predict_type='start'):
        """

        :param inputs_embed:
        :param input_lengths:
        :param predict_type: enum['first', 'start'].
                'first' means the first element of decoder input is the first element of encoder. In other word, it
                assumes the first token of the sequence will always true.
                'start' means the first element of decoder is <BEGIN> label.
        :return:
        """
        input_mask, position_input = self.do_input_position(inputs_embed, input_lengths)

        encode_value = self.do_encode(input_mask, position_input)

        batch_size = list(inputs_embed.shape)[0]
        continue_mask = transform_to_cuda(torch.Tensor([1 for i in range(batch_size)]).byte())
        output_stack = transform_to_cuda(torch.zeros(0, 0).long())
        predict_ids_record = []

        if predict_type == 'start':
            outputs = transform_to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)]))
        elif predict_type == 'first':
            outputs = inputs_embed[:, 0:1]
        for i in range(self.sequence_max_length):
            output_embed = self.output_embedding(outputs)

            # deal position encode
            if self.position_encode_fn is not None:
                dim_num = len(output_embed.shape) - 2
                position_start_list = [i for t in range(dim_num)]
                positioned_output = self.position_encode_fn(output_embed, position_start_list=position_start_list)
            else:
                positioned_output = output_embed

            one_output = self.do_decode(positioned_output, output_mask=None, encode_value=encode_value, input_mask=input_mask)
            output_value = self.output(one_output)

            # calculate and concat output
            softmax_output = F.softmax(output_value, dim=-1)
            one_predict_ids = (torch.topk(softmax_output, k=1, dim=-1)[1]).view(batch_size)
            one_predict_ids = torch.where(continue_mask, one_predict_ids, transform_to_cuda(torch.ones(batch_size).long()) * self.pad_label)
            # output_stack = torch.cat((output_stack, predict_ids), dim=1)
            # one_predict_ids shape: [batch_size]
            predict_ids_record = predict_ids_record + [one_predict_ids]

            # update batch status
            # predict_ids = torch.unsqueeze(one_predict_ids, dim=1)
            one_continue = torch.ne(one_predict_ids, self.end_label)
            continue_mask = continue_mask & one_continue

            if torch.sum(continue_mask) == 0:
                break

            # outputs = predict_ids
            outputs = torch.unsqueeze(one_predict_ids, dim=1)
        output_stack = torch.stack(predict_ids_record, dim=1)
        return output_stack


class OnlyAttentionFixErrorModel(nn.Module):

    def __init__(self, vocabulary_size, hidden_size, sequence_max_length, num_heads, start_label, end_label, pad_label, dropout_p=0.1, encoder_stack_num=1, decoder_stack_num=1, normalize_type='layer'):
        super(OnlyAttentionFixErrorModel, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.sequence_max_length = sequence_max_length
        self.start_label = start_label
        self.end_label = end_label
        self.pad_label = pad_label
        self.dropout_p = dropout_p

        self.dropout = transform_to_cuda(nn.Dropout(dropout_p))

        self.input_embedding = transform_to_cuda(nn.Embedding(vocabulary_size, hidden_size))

        self.position_encode_fn = add_position_encode

        if encoder_stack_num <= 0 or decoder_stack_num <= 0:
            raise Exception('stack_num should greater than 0: {}, {}'.format(encoder_stack_num, decoder_stack_num))

        self.seq2seq_without_input_embed = OnlyAttentionFixErrorModelWithoutInputEmbedding(
            vocabulary_size, hidden_size, sequence_max_length, num_heads, start_label, end_label, pad_label,
            dropout_p=dropout_p, encoder_stack_num=encoder_stack_num, decoder_stack_num=decoder_stack_num)

    # def do_input_embedding(self, inputs, input_lengths):
    #     # input embedding
    #     input_embed = self.dropout(self.input_embedding(inputs))
    #     if self.position_encode_fn is not None:
    #         position_input_embed = self.dropout(self.position_encode_fn(input_embed))
    #     else:
    #         position_input_embed = input_embed
    #     input_mask = create_sequence_length_mask(input_lengths, torch.max(input_lengths).item())
    #     return input_mask, input_embed

    def forward(self, inputs, input_lengths, outputs, output_lengths):
        # input_mask, position_input_embed = self.do_input_embedding(inputs, input_lengths)
        input_embed = self.dropout(self.input_embedding(inputs))

        output_value = self.seq2seq_without_input_embed.forward(input_embed, input_lengths, outputs, output_lengths)
        return output_value

    # def do_decode(self, position_output_embed, output_mask, encode_value, input_mask):
    #     # decoder for n times
    #     decoder_value = position_output_embed
    #     for decoder in self.decoder_list:
    #         decoder_value = decoder(decoder_value, output_mask, encode_value, input_mask)
    #     return decoder_value

    def forward_test(self, inputs, input_lengths, predict_type='start'):
        """

        :param inputs:
        :param input_lengths:
        :param predict_type: enum['first', 'start'].
                'first' means the first element of decoder input is the first element of encoder. In other word, it
                assumes the first token of the sequence will always true.
                'start' means the first element of decoder is <BEGIN> label.
        :return:
        """
        input_embed = self.dropout(self.input_embedding(inputs))
        output_stack = self.seq2seq_without_input_embed.forward_test(input_embed, input_lengths, predict_type)
        return output_stack
        # input_mask, position_input_embed = self.do_input_embedding(inputs, input_lengths)

        # encode_value = self.do_encode(input_mask, position_input_embed)

        # batch_size = list(inputs.shape)[0]
        # continue_mask = transform_to_cuda(torch.Tensor([1 for i in range(batch_size)]).byte())
        # output_stack = transform_to_cuda(torch.zeros(0, 0).long())
        #
        # if predict_type == 'start':
        #     outputs = transform_to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)]))
        # elif predict_type == 'first':
        #     outputs = inputs[:, 0:1]
        # for i in range(self.sequence_max_length):
        #     output_embed = self.output_embedding(outputs)
        #
        #     # deal position encode
        #     if self.position_encode_fn is not None:
        #         dim_num = len(output_embed.shape) - 2
        #         position_start_list = [i for t in range(dim_num)]
        #         output_embed = self.position_encode_fn(output_embed, position_start_list=position_start_list)
        #
        #     one_output = self.do_decode(output_embed, output_mask=None, encode_value=encode_value, input_mask=input_mask)
        #     output_value = self.output(one_output)
        #
        #     # calculate and concat output
        #     softmax_output = F.softmax(output_value, dim=-1)
        #     one_predict_ids = (torch.topk(softmax_output, k=1, dim=-1)[1]).view(batch_size)
        #     one_predict_ids = torch.where(continue_mask, one_predict_ids, transform_to_cuda(torch.ones(batch_size).long()) * self.pad_label)
        #     predict_ids = torch.unsqueeze(one_predict_ids, dim=1)
        #     output_stack = torch.cat((output_stack, predict_ids), dim=1)
        #
        #     # update batch status
        #     one_continue = torch.ne(predict_ids, self.end_label)
        #     continue_mask = continue_mask & one_continue.view(batch_size)
        #
        #     if torch.sum(continue_mask) == 0:
        #         break
        #
        #     outputs = predict_ids
        # return output_stack


def train(model, dataset, batch_size, loss_fn, optimizer, gap_token, begin_tokens, end_tokens, predict_type='start'):
    print('in train')
    model.train()
    total_loss = torch.Tensor([0])
    count = torch.Tensor([0])
    steps = 0

    begin_len = len(begin_tokens) if begin_tokens is not None else 0
    end_len = len(end_tokens) if end_tokens is not None else 0

    for data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        error_tokens = transform_to_cuda(torch.LongTensor(PaddedList(data['error_tokens'])))
        error_length = transform_to_cuda(torch.LongTensor(data['error_length']))
        ac_tokens_input = transform_to_cuda(torch.LongTensor(PaddedList(data['ac_tokens'])))
        ac_tokens_length = transform_to_cuda(torch.LongTensor(data['ac_length']))
        token_maps = transform_to_cuda(torch.LongTensor(PaddedList(data['token_map'], fill_value=TARGET_PAD_TOKEN)))

        model.zero_grad()

        # get split of error list. replace it to rl model
        stay_label_list = choose_token_random_batch(data['error_length'], data['error_mask'], random_value=0.2)


        part_tokens, part_ac_tokens = combine_spilt_tokens_batch(data['error_tokens'], data['ac_tokens'], stay_label_list, data['token_map'], gap_token, begin_tokens, end_tokens)
        print('part_tokens: length: {}/{},{}/{}'.format(len(part_tokens[0]), len(data['error_tokens'][0]), len(part_tokens[1]), len(data['error_tokens'][1])))
        if predict_type == 'start':
            encoder_input = part_tokens
            encoder_length = [len(inp) for inp in encoder_input]
            decoder_input = [tokens[:-1] for tokens in part_ac_tokens]
            decoder_length = [len(inp) for inp in decoder_input]
            target_output = [tokens[1:] for tokens in part_ac_tokens]

        elif predict_type == 'first':
            encoder_input = part_tokens
            encoder_length = [len(inp) for inp in encoder_input]
            decoder_input = [tokens[begin_len:-1] for tokens in part_ac_tokens]
            decoder_length = [len(inp) for inp in decoder_input]
            target_output = [tokens[begin_len+1:] for tokens in part_ac_tokens]

        encoder_input = transform_to_cuda(torch.LongTensor(PaddedList(encoder_input)))
        encoder_length = transform_to_cuda(torch.LongTensor(encoder_length))
        decoder_input = transform_to_cuda(torch.LongTensor(PaddedList(decoder_input)))
        decoder_length = transform_to_cuda(torch.LongTensor(decoder_length))
        target_output = PaddedList(target_output, fill_value=TARGET_PAD_TOKEN)

        log_probs = model.forward(encoder_input, encoder_length, decoder_input, decoder_length)
        loss = loss_fn(log_probs.view(-1, log_probs.shape[-1]), transform_to_cuda(torch.LongTensor(target_output)).view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        optimizer.step()

        cur_target_count = torch.sum(decoder_length.data.cpu()).float()
        total_loss += (loss.data.cpu() * cur_target_count)
        count += cur_target_count
        steps += 1

    return (total_loss/count).data.item()


def evaluate(model, dataset, batch_size, loss_fn, id_to_word_fn, file_path, gap_token, begin_tokens, end_tokens, predict_type, use_force_train=False):
    print('in evaluate')
    model.train()
    total_loss_in_train = torch.Tensor([0])
    count = torch.Tensor([0])
    count_in_train = torch.Tensor([0])
    steps = 0

    begin_len = len(begin_tokens) if begin_tokens is not None else 0
    end_len = len(end_tokens) if end_tokens is not None else 0

    for data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        error_tokens = transform_to_cuda(torch.LongTensor(PaddedList(data['error_tokens'])))
        error_length = transform_to_cuda(torch.LongTensor(data['error_length']))
        ac_tokens_input = transform_to_cuda(torch.LongTensor(PaddedList(data['ac_tokens'])))
        ac_tokens_length = transform_to_cuda(torch.LongTensor(data['ac_length']))
        token_maps = transform_to_cuda(torch.LongTensor(PaddedList(data['token_map'], fill_value=TARGET_PAD_TOKEN)))

        # get split of error list. replace it to rl model
        stay_label_list = choose_token_random_batch(data['error_length'], data['error_mask'], random_value=0.2)

        part_tokens, part_ac_tokens = combine_spilt_tokens_batch(data['error_tokens'], data['ac_tokens'], stay_label_list, data['token_map'], gap_token, begin_tokens, end_tokens)

        encoder_input = part_tokens
        encoder_length = [len(inp) for inp in encoder_input]

        if use_force_train:
            if predict_type == 'start':
                decoder_input = [tokens[:-1] for tokens in part_ac_tokens]
                decoder_length = [len(inp) for inp in decoder_input]
                target_output = [tokens[1:] for tokens in part_ac_tokens]
            elif predict_type == 'first':
                decoder_input = [tokens[begin_len:-1] for tokens in part_ac_tokens]
                decoder_length = [len(inp) for inp in decoder_input]
                target_output = [tokens[begin_len+1:] for tokens in part_ac_tokens]

            encoder_input = transform_to_cuda(torch.LongTensor(PaddedList(encoder_input)))
            encoder_length = transform_to_cuda(torch.LongTensor(encoder_length))
            decoder_input = transform_to_cuda(torch.LongTensor(PaddedList(decoder_input)))
            decoder_length = transform_to_cuda(torch.LongTensor(decoder_length))
            target_output = PaddedList(target_output, fill_value=TARGET_PAD_TOKEN)

            log_probs = model.forward(encoder_input, encoder_length, decoder_input, decoder_length)
            loss = loss_fn(log_probs.view(-1, log_probs.shape[-1]), transform_to_cuda(torch.LongTensor(target_output)).view(-1))

            cur_target_count = torch.sum(decoder_length.data.cpu()).float()
            total_loss_in_train += (loss.data.cpu() * cur_target_count)

            count_in_train += cur_target_count
        steps += 1

    return (total_loss_in_train/count_in_train).data.item()


def train_and_test(data_type, batch_size, hidden_size, num_heads, encoder_stack_num, decoder_stack_num, dropout_p, learning_rate, epoches, saved_name, load_name=None, gcc_file_path='test.c', normalize_type='layer', predict_type='start'):
    save_path = os.path.join(config.save_model_root, saved_name)
    if load_name is not None:
        load_path = os.path.join(config.save_model_root, load_name)

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
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name) for dd, name in zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = datasets
    print(train_dataset)
    model = OnlyAttentionFixErrorModel(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
                                       sequence_max_length=MAX_LENGTH, num_heads=num_heads,
                                       start_label=vocabulary.word_to_id(vocabulary.begin_tokens[0]),
                                       end_label=vocabulary.word_to_id(vocabulary.end_tokens[0]), pad_label=0,
                                       encoder_stack_num=encoder_stack_num, decoder_stack_num=decoder_stack_num,
                                       dropout_p=dropout_p, normalize_type=normalize_type)

    loss_function = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_TOKEN)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    best_valid_loss = None
    best_test_loss = None
    best_valid_compile_result = None
    best_test_compile_result = None
    if load_name is not None:
        torch_util.load_model(model, load_path)
        best_valid_loss = evaluate(model, valid_dataset, batch_size, loss_function, vocabulary.id_to_word, file_path=gcc_file_path, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type, use_force_train=True)
        best_test_loss = evaluate(model, test_dataset, batch_size, loss_function, vocabulary.id_to_word, file_path=gcc_file_path, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type, use_force_train=True)

    for epoch in range(epoches):
        train_loss = train(model, train_dataset, batch_size, loss_function, optimizer, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type)
        valid_loss = evaluate(model, valid_dataset, batch_size, loss_function, vocabulary.id_to_word, file_path=gcc_file_path, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type, use_force_train=True)
        # valid_compile_result = evaluate(model, valid_dataset, batch_size, loss_function, vocabulary.id_to_word, file_path=gcc_file_path, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type)
        test_loss = evaluate(model, test_dataset, batch_size, loss_function, vocabulary.id_to_word, file_path=gcc_file_path, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type, use_force_train=True)
        # test_compile_result = evaluate(model, test_dataset, batch_size, loss_function, vocabulary.id_to_word, file_path=gcc_file_path, gap_token=addition_tokens_id[0], begin_tokens=begin_tokens_id, end_tokens=end_tokens_id, predict_type=predict_type)

        valid_compile_result = 0
        test_compile_result = 0

        scheduler.step(valid_loss)

        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_compile_result = valid_compile_result
            best_test_loss = test_loss
            best_test_compile_result = test_compile_result
            if not is_debug:
                torch_util.save_model(model, save_path)
        print('epoch {}: train loss of {}, valid loss of {}, test loss of {}, '
              'valid compile result of {}, test compile result of {}'.format(epoch, train_loss, valid_loss,
                                                                             test_loss, valid_compile_result,
                                                                             test_compile_result))
    print('The model {} best valid loss of {}, best test loss of {}, best valid compile result of {}, '
          'best test compile result of {}'.format(saved_name, best_valid_loss, best_test_loss,
                                                  best_valid_compile_result, best_test_compile_result))


if __name__ == '__main__':

    train_and_test(data_type='', batch_size=2, hidden_size=512, num_heads=4, encoder_stack_num=2, decoder_stack_num=2,
                   dropout_p=0.2, learning_rate=0.001, epoches=2, saved_name='only_attention_fix_error_model_test.pkl',
                   load_name=None, gcc_file_path='test.c', normalize_type='layer', predict_type='start')



    # model = OnlyAttentionFixErrorModel(vocabulary_size=20, hidden_size=16, sequence_max_length=10, num_heads=2, start_label=0, end_label=1, pad_label=2, encoder_stack_num=2, decoder_stack_num=2, normalize_type='layer')
    #
    # inputs = [[0, 5, 7, 9, 12, 1],
    #           [0, 3, 16, 10, 1, 2]]
    # input_length = [6, 5]
    # output = [[0, 5, 7, 9, 1],
    #           [0, 3, 16, 1, 2]]
    # output_length = [5, 4]
    # inputs = torch.LongTensor(inputs)
    # input_length = torch.LongTensor(input_length)
    # output = torch.LongTensor(output)
    # output_length = torch.LongTensor(output_length)
    # res = model.forward(inputs, input_length, output, output_length)
    # print(res)
    # print(res.shape)
    #
    # res = model.forward_test(inputs, input_length)
    # print(res)
    # print(res.shape)










