import os
import random

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import more_itertools

import config
from common import torch_util, util
from common.args_util import to_cuda, get_model
from common.logger import init_a_file_logger, info
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from common.torch_util import create_sequence_length_mask, permute_last_dim_to_second, expand_tensor_sequence_len, \
    expand_tensor_sequence_to_same
from common.util import data_loader, show_process_map, PaddedList, convert_one_token_ids_to_code, filter_token_ids, \
    compile_c_code_by_gcc, create_token_set, create_token_mask_by_token_set, generate_mask, queued_data_loader
from experiment.experiment_util import load_common_error_data, load_common_error_data_sample_100, \
    create_addition_error_data, create_copy_addition_data, load_deepfix_error_data
from experiment.parse_xy_util import parse_output_and_position_map
from model.base_attention import TransformEncoderModel, TransformDecoderModel, TrueMaskedMultiHeaderAttention, \
    register_hook, PositionWiseFeedForwardNet, is_nan, record_is_nan
from read_data.load_data_vocabulary import create_common_error_vocabulary
from seq2seq.models import EncoderRNN, DecoderRNN
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK
from vocabulary.word_vocabulary import Vocabulary
from common.constants import pre_defined_c_tokens, pre_defined_c_library_tokens


MAX_LENGTH = 500
IGNORE_TOKEN = -1
is_debug = False


class CCodeErrorDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transform=None,
                 transformer_vocab_slk=None,
                 no_filter=False):
        self.set_type = set_type
        self.transform = transform
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df
            self._samples = [row for i, row in self.data_df.iterrows()]
            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)
            # for s in self._samples:
            #     for k, v in s.items():
            #         print("{}:shape {}".format(k, np.array(v).shape))

    def filter_df(self, df):
        df = df[df['error_code_word_id'].map(lambda x: x is not None)]
        # print('CCodeErrorDataSet df before: {}'.format(len(df)))
        df = df[df['distance'].map(lambda x: x >= 0)]
        # print('CCodeErrorDataSet df after: {}'.format(len(df)))
        # print(self.data_df['error_code_word_id'])
        # df['error_code_word_id'].map(lambda x: print(type(x)))
        # print(df['error_code_word_id'].map(lambda x: print(x))))
        df = df[df['error_code_word_id'].map(lambda x: len(x) < MAX_LENGTH)]
        end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])

        slk_count = 0
        error_count = 0
        not_in_mask_error = 0
        def filter_slk(ac_code_ids):
            res = None
            nonlocal slk_count, error_count, not_in_mask_error
            slk_count += 1
            if slk_count%100 == 0:
                print('slk count: {}, error count: {}, not in mask error: {}'.format(slk_count, error_count, not_in_mask_error))
            try:
                mask_list = self.transformer.get_all_token_mask_train([ac_code_ids])[0]
                res = True
                for one_id, one_mask in zip(ac_code_ids+[end_id], mask_list):
                    if one_id not in one_mask:
                        print('id {} not in mask {}'.format(one_id, one_mask))
                        not_in_mask_error += 1
                        res = None
            except Exception as e:
                error_count += 1
                # print('slk error : {}'.format(e))
                res = None
            return res

        print('before slk grammar: {}'.format(len(df)))
        df['grammar_mask_list'] = df['ac_code_word_id'].map(filter_slk)
        df = df[df['grammar_mask_list'].map(lambda x: x is not None)]
        print('after slk grammar: {}'.format(len(df)))

        return df

    def _get_raw_sample(self, row):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]

        sample = {"error_tokens": row['error_code_word_id'],
                  'error_length': len(row['error_code_word_id']),
                  'includes': row['includes']}
        if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['ac_tokens'] = [begin_id] + row['ac_code_word_id'] + [end_id]
            sample['ac_length'] = len(sample['ac_tokens'])
            # sample['token_map'] = self.data_df.iloc[index]['token_map']
            # sample['pointer_map'] = create_pointer_map(sample['ac_length'], sample['token_map'])
            sample['pointer_map'] = [0] + row['pointer_map'] + [sample['error_length']-1]
            # sample['error_mask'] = self.data_df.iloc[index]['error_mask']
            sample['is_copy'] = [0] + row['is_copy'] + [0]
            sample['distance'] = row['distance']
            # sample['grammar_mask_list'] = row['grammar_mask_list']
            sample['grammar_mask_list'] = self.transformer.get_all_token_mask_train([row['ac_code_word_id']])[0]
            sample['grammar_mask_length'] = [len(ma) for ma in sample['grammar_mask_list']]
            # sample['grammar_mask_index_to_id_dict'] = [{i: m for i, m in enumerate(masks)} for masks in row['grammar_mask_list']]
            # sample['grammar_mask_id_to_index_dict'] = [{m: i for i, m in enumerate(masks)} for masks in row['grammar_mask_list']]
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            # sample['token_map'] = None
            sample['pointer_map'] = None
            # sample['error_mask'] = None
            sample['is_copy'] = None
            sample['distance'] = 0
            sample['grammar_mask_list'] = None
            sample['grammar_mask_length'] = None
            # sample['grammar_mask_index_to_id_dict'] = None
            # sample['grammar_mask_id_to_index_dict'] = None
        return sample

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [row for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = CCodeErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type, transform=self.transform,
                              transformer_vocab_slk=self.transformer)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = CCodeErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                              transform=self.transform, transformer_vocab_slk=self.transformer)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __len__(self):
        return len(self._samples)


def load_dataset(is_debug, vocabulary, mask_transformer):
    if is_debug:
        data_dict = load_common_error_data_sample_100()
    else:
        data_dict = load_common_error_data()
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name, transformer_vocab_slk=mask_transformer) for
                dd, name in
                zip(data_dict, ["train", "all_valid", "all_test"])]
    multi_step_no_target = False
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets

    ac_copy_data_dict = create_copy_addition_data(train_dataset.data_df['ac_code_word_id'],
                                                  train_dataset.data_df['includes'])
    ac_copy_dataset = CCodeErrorDataSet(pd.DataFrame(ac_copy_data_dict), vocabulary, 'ac_copy',
                                        transformer_vocab_slk=mask_transformer, no_filter=True)
    return train_dataset, valid_dataset, test_dataset, ac_copy_dataset


class DynamicFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weight = to_cuda(nn.Embedding(output_size, hidden_size))
        self.bias = to_cuda(nn.Parameter(torch.rand(output_size)))
        self.relu = nn.ReLU()

    def forward(self, input_value, mask_tensor):
        """

        :param input_value: [batch, seq, hidden_size]
        :param mask_tensor: [batch, seq, mask_set]
        :param mask_mask: [batcg, seq, mask_set]
        :return:
        """
        combine_shape = list(input_value.shape[:-1])

        mask_tensor_shape = mask_tensor.shape
        # [batch, seq, mask, hidden]
        part_weight = self.weight(mask_tensor)
        # [batch, seq, mask]
        part_bias = torch.index_select(self.bias, dim=0, index=mask_tensor.view(-1)).view(mask_tensor_shape)

        # [-1, hidden, mask]
        part_weight = part_weight.view(-1, part_weight.shape[-2], part_weight.shape[-1]).permute(0, 2, 1)
        # [-1, mask]
        part_bias = part_bias.view(-1, part_bias.shape[-1])

        input_value = input_value.view(-1, 1, input_value.shape[-1])
        output_value = torch.bmm(input_value, part_weight).view(input_value.shape[0], mask_tensor_shape[-1]) + part_bias
        output_value = output_value.view(*combine_shape, mask_tensor_shape[-1])
        output_value = self.relu(output_value)
        return output_value


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
            position_input_index.masked_fill_(~input_mask, MAX_LENGTH)
        position_input_embedded = self.position_embedding(position_input_index)
        position_input = torch.cat([position_input_embedded, embedded_input], dim=-1)
        return position_input

    def forward_position(self, inputs, input_mask=None):
        batch_size = inputs.shape[0]
        input_sequence_len = inputs.shape[1]
        position_input_index = to_cuda(
            torch.unsqueeze(torch.arange(0, input_sequence_len), dim=0).expand(batch_size, -1)).long()
        if input_mask is not None:
            position_input_index.masked_fill_(~input_mask, MAX_LENGTH)
        position_input_embedded = self.position_embedding(position_input_index)
        return position_input_embedded

    def forward_content(self, inputs, input_mask=None):
        embedded_input = self.embedding(inputs)
        return embedded_input


class RNNPointerNetworkModelWithSLKMask(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_layers, start_label, end_label, dropout_p=0.1, MAX_LENGTH=500,
                 atte_position_type='position', mask_transformer: TransformVocabularyAndSLK=None):
        super(RNNPointerNetworkModelWithSLKMask, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.start_label = start_label
        self.end_label = end_label
        self.dropout_p = dropout_p
        self.MAX_LENGTH = MAX_LENGTH
        self.atte_position_type = atte_position_type

        self.mask_transformer = mask_transformer


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
        # self.output_linear = PositionWiseFeedForwardNet(input_size=self.hidden_size, hidden_size=self.hidden_size,
        #                                                 output_size=vocabulary_size, hidden_layer_count=1)

        # self.output_linear = DynamicFeedForward(input_size=self.hidden_size, hidden_size=self.hidden_size,
        #                                         output_size=vocabulary_size)
        self.dynamic_output_linear = DynamicFeedForward(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                                output_size=vocabulary_size)

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

    def decode_step(self, decode_input, encoder_hidden, decode_mask, encode_value, encode_mask, position_embedded,
                    teacher_forcing_ratio=1, value_mask=None, value_set=None):
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

        # value_output = self.output_linear(decoder_output, value_set, value_mask)
        # print('before dynamic linear')
        # value_output = self.output_linear(decoder_output)
        value_output = self.dynamic_output_linear(decoder_output, value_set)
        # print('after dynamic linear')
        if value_mask is not None:
            dim_len = len(value_output.shape)
            value_output.masked_fill_(~value_mask.view(value_mask.shape[0], *[1 for i in range(dim_len - 3)],
                                                            value_mask.shape[-2], value_mask.shape[-1]), -float('inf'))
        if decode_mask is not None:
            value_output = value_output.masked_fill_(~decode_mask.view(decode_mask.shape[0], decode_mask.shape[1], *[1 for i in range(len(value_output.shape)-2)]), 0)
        return is_copy, value_output, pointer_output, hidden

    def forward(self, inputs, input_mask, output, output_mask, value_mask_set_tensor, mask_mask=None, test=False):
        if test:
            return self.forward_test(inputs, input_mask, value_mask=mask_mask)

        # output_length = to_cuda(torch.sum(output_mask, dim=-1))
        # output_token_length = output_length - 2
        # print('vocabulary size: ', self.vocabulary_size)
        # python list [batch, seq, mask_list]
        # value_mask_set = self.mask_transformer.get_all_token_mask_train(ac_tokens_ids=output_list)
        # value_mask_set_len = [[len(ma) for ma in seq_mask] for seq_mask in value_mask_set]
        # mask_length = to_cuda(torch.LongTensor(PaddedList(value_mask_set_len))).view(-1)
        # mask_mask = create_sequence_length_mask(mask_length).view(output.shape[0], output.shape[1]-1, -1)
        # value_mask_set_tensor = to_cuda(torch.LongTensor(PaddedList(value_mask_set)))
        # print('value_mask_set shape : {}'.format(value_mask_set_tensor.shape))

        # print('after create tensor')
        # np_mask = np.array(paddded_mask)
        # value_mask = to_cuda(torch.ByteTensor(PaddedList(value_mask, shape=[output.shape[0], output.shape[1]-1, self.vocabulary_size])))
        # value_mask = None

        position_input = self.encode_position_embedding(inputs, input_mask)
        if self.atte_position_type == 'position':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
        elif self.atte_position_type == 'content':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_content(inputs, input_mask)
        else:
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(inputs, input_mask)
        encode_value, hidden = self.encoder(position_input)
        encoder_hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        position_output = self.decode_position_embedding(inputs=output, input_mask=output_mask)
        is_copy, value_output, pointer_output, _ = self.decode_step(decode_input=position_output, encoder_hidden=encoder_hidden,
                                                                 decode_mask=output_mask, encode_value=encode_value,
                                                                 position_embedded=encoder_atte_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=1,
                                                                    value_set=value_mask_set_tensor, value_mask=mask_mask)
        return is_copy, value_output, pointer_output

    def forward_test(self, input, input_mask, value_mask):
        position_input = self.encode_position_embedding(input, input_mask)
        if self.atte_position_type == 'position':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
        elif self.atte_position_type == 'content':
            encoder_atte_input_embedded = self.encode_position_embedding.forward_content(input, input_mask)
        else:
            encoder_atte_input_embedded = self.encode_position_embedding.forward_position(input, input_mask)
        encode_value, hidden = self.encoder(position_input)
        hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in hidden]

        batch_size = list(input.shape)[0]
        continue_mask = to_cuda(torch.Tensor([1 for i in range(batch_size)])).byte()
        outputs = to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)]))
        is_copy_stack = []
        value_output_stack = []
        pointer_stack = []
        output_stack = []

        slk_iterator_list = [self.mask_transformer.create_new_slk_iterator() for i in range(batch_size)]


        for i in range(self.MAX_LENGTH):
            mask_list = [self.mask_transformer.get_candicate_step(iter) for iter in slk_iterator_list]


            cur_input_mask = continue_mask.view(continue_mask.shape[0], 1)
            output_embed = self.decode_position_embedding(inputs=outputs, input_mask=cur_input_mask, start_index=i)

            is_copy, value_output, pointer_output, hidden = self.decode_step(decode_input=output_embed, encoder_hidden=hidden,
                                                                 decode_mask=cur_input_mask, encode_value=encode_value,
                                                                 position_embedded=encoder_atte_input_embedded,
                                                                 encode_mask=input_mask, teacher_forcing_ratio=0,
                                                                             value_mask=value_mask)
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
        return is_copy_result, value_result, pointer_result


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


def create_parse_rnn_input_batch_data_fn(vocab):

    def parse_rnn_input_batch_data(batch_data, do_sample=False, add_value_mask=False):
        error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
        max_input = max(batch_data['error_length'])
        input_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['error_length'])), max_len=max_input))

        if not do_sample:
            # ac_tokens_decoder_input = [bat[:-1] for bat in batch_data['ac_tokens']]
            ac_tokens_decoder_input = batch_data['ac_tokens']
            ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_input)))
            ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_input]
            max_output = max(ac_tokens_length)
            output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))
        else:
            ac_tokens = None
            output_mask = None

        if add_value_mask:
            token_sets = [create_token_set(one, vocab) for one in batch_data['error_tokens']]
            if not do_sample:
                ac_token_sets = [create_token_set(one, vocab) for one in batch_data['ac_tokens']]
                token_sets = [tok_s | ac_tok_s for tok_s, ac_tok_s in zip(token_sets, ac_token_sets)]
            token_mask = [create_token_mask_by_token_set(one, vocab.vocabulary_size) for one in token_sets]
            token_mask_tensor = to_cuda(torch.ByteTensor(token_mask))
        else:
            token_mask_tensor = None

        batch_size = len(batch_data['error_tokens'])

        # value_ac_tokens = batch_data['ac_tokens']
        # value_ac_tokens = [tokens[1:-1] for tokens in value_ac_tokens]
        # value_mask_list = transformer.get_all_token_mask_train(value_ac_tokens)
        # value_mask_set_len = [[len(ma) for ma in mask_list] for mask_list in value_mask_list]
        # batch_data['grammar_mask_list'] = value_mask_list

        value_mask_list = batch_data['grammar_mask_list']
        value_mask_set_len = batch_data['grammar_mask_length']
        max_seq = max([len(s) for s in value_mask_list])
        max_mask_len = max(more_itertools.collapse(value_mask_set_len))
        value_mask_set_tensor = to_cuda(torch.LongTensor(PaddedList(value_mask_list, shape=[batch_size, max_seq, max_mask_len])))
        # [batch, seq, mask_len]
        mask_length = to_cuda(torch.LongTensor(PaddedList(value_mask_set_len, shape=[batch_size, max_seq])))
        mask_length_shape = list(mask_length.shape)
        mask_mask = create_sequence_length_mask(mask_length.view(-1)).view(mask_length_shape[0], mask_length_shape[1], -1)

        # return error_tokens, input_mask, ac_tokens, output_mask
        return error_tokens, input_mask, ac_tokens, output_mask, value_mask_set_tensor, mask_mask
    return parse_rnn_input_batch_data


def create_parse_target_batch_data():

    def parse_target_batch_data(batch_data):
        ac_tokens_decoder_output = [bat[1:] for bat in batch_data['ac_tokens']]
        target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_decoder_output, fill_value=IGNORE_TOKEN)))
        grammar_id_to_index_dict = [[{m: i for i, m in enumerate(masks)} for masks in grammar_mask_list]
                                    for grammar_mask_list in batch_data['grammar_mask_list']]
        ac_tokens_value_output = [[di[tok] for tok, di in zip(ac_tokens, value_mask_dict)]
                                    for ac_tokens, value_mask_dict in
                                    zip(ac_tokens_decoder_output, grammar_id_to_index_dict)]
        target_value_tokens = to_cuda(torch.LongTensor(PaddedList(ac_tokens_value_output, fill_value=IGNORE_TOKEN)))

        ac_tokens_length = [len(bat) for bat in ac_tokens_decoder_output]
        max_output = max(ac_tokens_length)
        output_mask = to_cuda(create_sequence_length_mask(to_cuda(torch.LongTensor(ac_tokens_length)), max_len=max_output))

        pointer_map = [bat[1:] for bat in batch_data['pointer_map']]
        target_pointer_output = to_cuda(torch.LongTensor(PaddedList(pointer_map, fill_value=IGNORE_TOKEN)))

        is_copy = [bat[1:] for bat in batch_data['is_copy']]
        target_is_copy = to_cuda(torch.Tensor(PaddedList(is_copy)))

        return target_is_copy, target_pointer_output, target_value_tokens, target_ac_tokens, output_mask
    return parse_target_batch_data

def create_combine_loss_fn(copy_weight=1, pointer_weight=1, value_weight=1, average_value=True):
    copy_loss_fn = nn.BCELoss(reduce=False)
    pointer_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)
    value_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)

    def combine_total_loss(is_copy, value_log_probs, pointer_log_probs, target_copy, target_pointer, target_value,
                           target_ac_token, output_mask):
        # record_is_nan(is_copy, 'is_copy: ')
        # record_is_nan(pointer_log_probs, 'pointer_log_probs: ')
        # record_is_nan(value_log_probs, 'value_log_probs: ')
        output_mask_float = output_mask.float()
        # info('output_mask_float: {}, mask batch: {}'.format(str(output_mask_float), torch.sum(output_mask_float, dim=-1)))
        # info('target_copy: {}, mask batch: {}'.format(str(target_copy), torch.sum(target_copy, dim=-1)))


        no_copy_weight = torch.where(target_copy.byte(), target_copy, to_cuda(torch.Tensor([15]).view(*[1 for i in range(len(target_copy.shape))])))
        copy_loss = copy_loss_fn(is_copy, target_copy)
        # record_is_nan(copy_loss, 'copy_loss: ')
        copy_loss = copy_loss * output_mask_float * no_copy_weight
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


def create_output_ids(model_output, model_input):
    """

    :param copy_output: [batch, sequence, dim**]
    :param value_output: [batch, sequence, dim**, vocabulary_size]
    :param pointer_output: [batch, sequence, dim**, encode_length]
    :param input: [batch, encode_length, dim**]
    :return: [batch, sequence, dim**]
    """
    is_copy, value_output, pointer_output = model_output
    error_tokens = model_input[0]
    mask_tensor = model_input[4]
    is_copy = (is_copy > 0.5)
    _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
    top_id = torch.gather(mask_tensor, dim=-1, index=top_id)
    _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
    pointer_pos_in_input = pointer_pos_in_input.squeeze(dim=-1)
    point_id = torch.gather(error_tokens, dim=-1, index=pointer_pos_in_input)
    next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
    return next_output


def slk_expand_output_and_target(model_output, model_target):
    model_output = list(model_output)
    model_target = list(model_target)
    model_output[0], model_target[0] = expand_tensor_sequence_to_same(model_output[0], model_target[0], fill_value=0)
    model_output[1], model_target[1] = expand_tensor_sequence_to_same(model_output[1], model_target[1], fill_value=IGNORE_TOKEN)
    model_output[2], model_target[2] = expand_tensor_sequence_to_same(model_output[2], model_target[2], fill_value=IGNORE_TOKEN)

    expand_len = model_output[2].shape[1]
    model_target[3] = expand_tensor_sequence_len(model_target[3], max_len=expand_len, fill_value=IGNORE_TOKEN)
    model_target[4] = expand_tensor_sequence_len(model_target[4], max_len=expand_len, fill_value=0)
    return model_output, model_target


