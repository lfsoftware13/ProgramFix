import torch
import torch.nn as nn
import torch.autograd as autograd
import config
import os
import more_itertools

from common.problem_util import to_cuda
from common.torch_util import calculate_accuracy_of_code_completion, get_predict_and_target_tokens, \
    remove_last_item_in_sequence, reverse_tensor
from common.util import batch_holder, transform_id_to_token, PaddedList, show_process_map, CustomerDataSet
from common import util, torch_util
from sklearn.utils import shuffle
import sys

from seq2seq.models import EncoderRNN
import pandas as pd

from vocabulary.word_vocabulary import Vocabulary

gpu_index = 0
BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]


class SensibilityRNNDataset(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transformer_vocab_slk=None,
                 no_filter=False,
                 do_flatten=False,
                 MAX_LENGTH=500,
                 use_ast=False,
                 do_multi_step_sample=False,
                 id_to_program_dict=None,
                 no_id_to_program_dict=False):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        self.is_flatten = do_flatten
        self.max_length = MAX_LENGTH
        self.use_ast = use_ast
        self.transform = False
        self.do_multi_step_sample = do_multi_step_sample
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df

            self.only_first = do_multi_step_sample
            from experiment.experiment_dataset import FlattenRandomIterateRecords
            self._samples = [FlattenRandomIterateRecords(row, is_flatten=do_flatten, only_first=do_multi_step_sample)
                             for i, row in self.data_df.iterrows()]
            # c = 0
            # for i, (index, row) in self.data_df.iterrows():
            #     print(i)
            #     print(row['id'])
            self.program_to_position_dict = {row['id']: i for i, (index, row) in enumerate(self.data_df.iterrows())}

            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)
            # for s in self._samples:
            #     for k, v in s.items():
            #         print("{}:shape {}".format(k, np.array(v).shape))

    def filter_df(self, df):
        df = df[df['error_token_id_list'].map(lambda x: x is not None)]
        df = df[df['distance'].map(lambda x: x >= 0)]
        def iterate_check_max_len(x):
            if not self.is_flatten:
                for i in x:
                    if len(i) > self.max_length:
                        return False
                return True
            else:
                return len(x) < self.max_length
        df = df[df['error_token_id_list'].map(iterate_check_max_len)]

        return df

    def set_only_first(self, only_first):
        self.only_first = only_first

    def _get_raw_sample(self, row):
        # sample = dict(row)
        row.select_random_i(only_first=self.only_first)
        sample = {}
        sample['id'] = row['id']
        sample['includes'] = row['includes']
        # if not self.is_flatten and self.do_multi_step_sample:
        #     sample['input_seq'] = row['error_token_id_list'][0]
        #     sample['input_seq_name'] = row['error_token_name_list'][0][1:-1]
        #     sample['input_length'] = len(sample['input_seq'])
        # elif not self.is_flatten and not self.do_multi_step_sample:
        #     sample['input_seq'] = row['error_token_id_list']
        #     sample['input_seq_name'] = [r[1:-1] for r in row['error_token_name_list']]
        #     sample['input_length'] = [len(ids) for ids in sample['input_seq']]
        # else:
        sample['input_seq'] = row['error_token_id_list']
        sample['input_seq_name'] = row['error_token_name_list'][1:-1]
        sample['input_length'] = len(sample['input_seq'])
        sample['copy_length'] = sample['input_length']
        sample['includes'] = row['includes']
        sample['distance'] = row['distance']

        sample['forward_target'] = row['error_token_id_list'][1:]
        sample['backward_target'] = row['error_token_id_list'][:-1]
        return sample

    def __getitem__(self, index):
        real_position = index
        row = self._samples[real_position]
        return self._get_raw_sample(row)

    def __setitem__(self, key, value):
        real_position = key
        self._samples[real_position] = value

    def __len__(self):
        return len(self._samples)


class LSTMModel(nn.Module):

    def __init__(self, dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional=False, dropout=0):
        super(LSTMModel, self).__init__()
        self.dictionary_size = dictionary_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.drop = nn.Dropout(dropout)

        print('dictionary_size: {}, embedding_dim: {}, hidden_size: {}, num_layers: {}, batch_size: {}, bidirectional: {}, dropout: {}'.format(
            dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional, dropout))

        self.bidirectional_num = 2 if bidirectional else 1

        print('before create embedding')
        self.word_embeddings = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=embedding_dim, padding_idx=0)
        print('before create lstm')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout)
        print('before create tag')
        self.hidden2tag = nn.Linear(hidden_size * self.bidirectional_num, dictionary_size)

        print('before init hidden')
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, cur_batch_size):
        return (to_cuda(torch.randn(self.num_layers * self.bidirectional_num, cur_batch_size, self.hidden_size)),
                to_cuda(torch.randn(self.num_layers * self.bidirectional_num, cur_batch_size, self.hidden_size)))

    def forward(self, inputs, token_lengths):
        """
        inputs: [batch_size, code_length]
        token_lengths: [batch_size, ]
        :param inputs:
        :return:
        """
        self.hidden = self.init_hidden(inputs.shape[0])
        # inputs = torch.LongTensor(inputs)
        # token_lengths = torch.LongTensor(token_lengths)

        cur_batch_size = len(inputs)

        _, idx_sort = torch.sort(token_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        inputs = torch.index_select(inputs, 0, idx_sort)
        token_lengths = list(torch.index_select(token_lengths, 0, idx_sort))

        print('input_size: ', inputs.size())

        embeds = self.word_embeddings(inputs).view(cur_batch_size, -1, self.embedding_dim)
        # print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        embeds = embeds.view(cur_batch_size, -1, self.embedding_dim)
        embeds = self.drop(embeds)
        # print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        # print('embeds value: {}'.format(embeds.data))
        # print('after embeds token_length: {}'.format(token_lengths))
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(embeds, token_lengths, batch_first=True)
        # print('packed_inputs batch size: ', len(packed_inputs.batch_sizes))
        # print('packed_inputs is cuda: {}'.format(packed_inputs.data.is_cuda))
        lstm_out, self.hidden = self.lstm(packed_inputs, self.hidden)

        unpacked_lstm_out, unpacked_lstm_length = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                                               padding_value=0)
        unpacked_lstm_out = self.drop(unpacked_lstm_out)
        dict_output = self.hidden2tag(unpacked_lstm_out)
        packed_dict_output = torch.nn.utils.rnn.pack_padded_sequence(dict_output, token_lengths, batch_first=True)


        # print('lstm_out batch size: ', len(lstm_out.batch_sizes))
        # print('lstm_out is cuda: ', lstm_out.data.is_cuda)
        # print('lstm value: {}'.format(lstm_out.data))

        # packed_output = nn.utils.rnn.PackedSequence(self.hidden2tag(lstm_out.data).cuda(gpu_index), lstm_out.batch_sizes)    # output shape: [batch_size, token_length, dictionary_size]
        # print('packed_output batch size: ', len(packed_output.batch_sizes))
        # print('packed_output is cuda: ', packed_output.data.is_cuda)

        unpacked_out, unpacked_length = torch.nn.utils.rnn.pad_packed_sequence(packed_dict_output, batch_first=True, padding_value=0)
        # print('unpacked_out: {}, unpacked_length: {}'.format(unpacked_out.size(), unpacked_length))
        unpacked_out = torch.index_select(unpacked_out, 0, torch.Tensor(idx_unsort).to(inputs.device))
        # print('unsort unpacked_out: {}'.format(unpacked_out.size()))
        # print('unsort unpacked_out is cuda: {}'.format(unpacked_out.is_cuda))

        return unpacked_out


class SplitRNNModelWarpper(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, encoder_params):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(hidden_size=hidden_size, **encoder_params)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_seq, input_length):
        encoder_output, _ = self.encoder(input_seq)
        split_encoder_output = remove_last_item_in_sequence(encoder_output, input_length, k=1)
        o = self.output(split_encoder_output)
        return o


class SensibilityBiRnnModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_size, encoder_params):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim, padding_idx=0)
        self.forward_rnn = SplitRNNModelWarpper(vocabulary_size, hidden_size, encoder_params)
        self.backward_rnn = SplitRNNModelWarpper(vocabulary_size, hidden_size, encoder_params)

    def forward(self, input_seq, input_length):
        backward_input_seq = reverse_tensor(input_seq, input_length)
        embedded_forward_input_seq = self.embedding(input_seq)
        embedded_backward_input_seq = self.embedding(backward_input_seq)
        forward_output = self.forward_rnn(embedded_forward_input_seq, input_length)
        backward_output = self.backward_rnn(embedded_backward_input_seq, input_length)
        reversed_backward_output = reverse_tensor(backward_output, input_length-1)
        return forward_output, reversed_backward_output


def create_loss_function(ignore_id):
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
    def loss_fn(forward_output, backward_output,
                forward_target_seq, backward_target_seq):
        forward_loss = cross_loss(forward_output.permute(0, 2, 1), forward_target_seq)
        backward_loss = cross_loss(backward_output.permute(0, 2, 1), backward_target_seq)
        total_loss = forward_loss + backward_loss
        return total_loss
    return loss_fn


def rnn_parse_input_batch_data_fn():
    def parse_input_batch_data(batch_data, do_sample=False):
        def to_long(x):
            return to_cuda(torch.LongTensor(x))
        input_seq = to_long(PaddedList(batch_data['input_seq']))
        input_length = to_long(batch_data['input_length'])
        return input_seq, input_length
    return parse_input_batch_data


def rnn_parse_target_batch_data_fn(ignore_id):
    def parse_target_batch_data(batch_data, ):
        forward_target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['forward_target'], fill_value=ignore_id)))
        backward_target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['backward_target'], fill_value=ignore_id)))
        return forward_target_seq, backward_target_seq
    return parse_target_batch_data


def create_output_fn(*args, **kwargs):
    pass





