import copy

import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import config
import os
import more_itertools

from model.sensibility_baseline.dual_lstm import DualLSTMModelWrapper
from model.sensibility_baseline.fix import LSTMFixerUpper
from model.sensibility_baseline.rnn_pytorch import SensibilityBiRnnModel
from model.sensibility_baseline.utility_class import Edit
from vocabulary.word_vocabulary import Vocabulary

def generate_action_from_edit(edit: Edit, length, empty_id):
    if edit.code == 'i':
        #insert before
        index = edit.index
        token = edit.token
        if index == 0:
            index = 1
        return index-1, index, token

    elif edit.code == 's':
        #substitute here
        index = edit.index
        token = edit.token
        if index == 0:
            index += 1
        elif index == length - 1:
            index -= 1
        return index-1, index+1, token

    elif edit.code == 'x':
        #delete here
        index = edit.index
        if index == 0:
            return index, index+1, empty_id
        elif index == length - 1:
            return index-1, index, empty_id
        else:
            return index-1, index+1, empty_id

class FixModel(nn.Module):
    def __init__(self,
                 rnn_model: SensibilityBiRnnModel,
                 vocabulary: Vocabulary):
        super().__init__()
        self.rnn_model = rnn_model
        self.dual_rnn_model = DualLSTMModelWrapper(rnn_model)
        self.fix_upper = LSTMFixerUpper(self.dual_rnn_model, vocabulary, 1)
        self.vocabulary = vocabulary
        self.empty_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
        self.tokens_set = torch.range(0, vocabulary.vocabulary_size).long().reshape(1, -1)

    def forward(self,
                batch_data, do_sample=True, do_beam_search=False,
                ):
        output = []
        ori_input_seq = batch_data['input_seq']
        max_length = max(len(t) for t in ori_input_seq)
        for seq in ori_input_seq:
            output.append(self.fix_upper.fix(seq))

        ori_batch_data = copy.copy(batch_data)

        for k in ori_batch_data.keys():
            t = []
            for i, l in enumerate(output):
                t.extend([ori_batch_data[k][i]]*len(l))
            batch_data[k] = t

        output = list(more_itertools.flatten(output))
        batch_size = len(output)

        p1 = torch.zeros((batch_size, max_length),)
        p2 = torch.zeros((batch_size, max_length),)
        is_copy = torch.ones((batch_size, 1)) * (-1000)
        copy_output = torch.zeros((batch_size, 1, max_length))
        sample_output = torch.zeros((batch_size, 1, self.vocabulary.vocabulary_size))

        for i, o in enumerate(output):
            p1_t, p2_t, s_t = generate_action_from_edit(o, len(batch_data['input_seq'][i]), self.empty_id)
            p1[i, p1_t] = 1.0
            p2[i, p2_t] = 1.0
            sample_output[i, 0, s_t] = 1.0

        return p1, p2, is_copy, copy_output, sample_output, self.tokens_set.expand(batch_size, -1).unsqueeze(1)


def parse_input_batch_data_fn(batch_data, do_sample):
    return [batch_data]

