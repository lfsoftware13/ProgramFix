import torch
import torch.nn as nn
import torch.autograd as autograd
import config
import os
import more_itertools

from common.torch_util import calculate_accuracy_of_code_completion, get_predict_and_target_tokens
from common.util import batch_holder, transform_id_to_token
from common import util, torch_util
from sklearn.utils import shuffle
import sys

gpu_index = 0
BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]


class LSTMModel(nn.Module):

    def __init__(self, dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional=False, dropout=0):
        super(LSTMModel, self).__init__()
        self.dictionary_size = dictionary_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.drop = nn.Dropout(dropout).cuda(gpu_index)

        print('dictionary_size: {}, embedding_dim: {}, hidden_size: {}, num_layers: {}, batch_size: {}, bidirectional: {}, dropout: {}'.format(
            dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional, dropout))

        self.bidirectional_num = 2 if bidirectional else 1

        print('before create embedding')
        self.word_embeddings = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=embedding_dim, padding_idx=0).cuda(gpu_index)
        print('before create lstm')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout).cuda(gpu_index)
        print('before create tag')
        self.hidden2tag = nn.Linear(hidden_size * self.bidirectional_num, dictionary_size).cuda(gpu_index)

        print('before init hidden')
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, cur_batch_size):
        return (autograd.Variable(torch.randn(self.num_layers * self.bidirectional_num, cur_batch_size, self.hidden_size)).cuda(gpu_index),
                autograd.Variable(torch.randn(self.num_layers * self.bidirectional_num, cur_batch_size, self.hidden_size)).cuda(gpu_index))

    def forward(self, inputs, token_lengths):
        """
        inputs: [batch_size, code_length]
        token_lengths: [batch_size, ]
        :param inputs:
        :return:
        """
        # hidden = self.init_hidden()
        inputs = torch.LongTensor(inputs)
        token_lengths = torch.LongTensor(token_lengths)

        cur_batch_size = len(inputs)

        _, idx_sort = torch.sort(token_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        inputs = torch.index_select(inputs, 0, idx_sort)
        token_lengths = list(torch.index_select(token_lengths, 0, idx_sort))

        print('input_size: ', inputs.size())

        embeds = self.word_embeddings(autograd.Variable(inputs).cuda(gpu_index)).view(cur_batch_size, -1, self.embedding_dim).cuda(gpu_index)
        # print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        embeds = embeds.view(cur_batch_size, -1, self.embedding_dim)
        embeds = self.drop(embeds).cuda(gpu_index)
        print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        # print('embeds value: {}'.format(embeds.data))
        # print('after embeds token_length: {}'.format(token_lengths))
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(embeds, token_lengths, batch_first=True)
        # print('packed_inputs batch size: ', len(packed_inputs.batch_sizes))
        # print('packed_inputs is cuda: {}'.format(packed_inputs.data.is_cuda))
        lstm_out, self.hidden = self.lstm(packed_inputs, self.hidden)

        unpacked_lstm_out, unpacked_lstm_length = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                                               padding_value=0)
        unpacked_lstm_out = self.drop(unpacked_lstm_out)
        dict_output = self.hidden2tag(unpacked_lstm_out).cuda(gpu_index)
        packed_dict_output = torch.nn.utils.rnn.pack_padded_sequence(dict_output, token_lengths, batch_first=True)


        # print('lstm_out batch size: ', len(lstm_out.batch_sizes))
        # print('lstm_out is cuda: ', lstm_out.data.is_cuda)
        # print('lstm value: {}'.format(lstm_out.data))

        # packed_output = nn.utils.rnn.PackedSequence(self.hidden2tag(lstm_out.data).cuda(gpu_index), lstm_out.batch_sizes)    # output shape: [batch_size, token_length, dictionary_size]
        # print('packed_output batch size: ', len(packed_output.batch_sizes))
        # print('packed_output is cuda: ', packed_output.data.is_cuda)

        unpacked_out, unpacked_length = torch.nn.utils.rnn.pad_packed_sequence(packed_dict_output, batch_first=True, padding_value=0)
        # print('unpacked_out: {}, unpacked_length: {}'.format(unpacked_out.size(), unpacked_length))
        unpacked_out = torch.index_select(unpacked_out, 0, autograd.Variable(idx_unsort).cuda(gpu_index))
        # print('unsort unpacked_out: {}'.format(unpacked_out.size()))
        # print('unsort unpacked_out is cuda: {}'.format(unpacked_out.is_cuda))

        return unpacked_out



