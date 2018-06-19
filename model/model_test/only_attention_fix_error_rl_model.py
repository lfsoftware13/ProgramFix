import math
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd

from torch.distributions import Categorical

from common.torch_util import create_sequence_length_mask
from common.util import data_loader, PaddedList
from model.model_test.only_attention_fix_error_model import add_position_encode, MaskedMultiHeaderAttention, \
    PositionWiseFeedForwardNet, SelfAttentionEncoder


is_cuda = False
gpu_index = 1

def transform_to_cuda(x):
    if is_cuda:
        x = x.cuda(gpu_index)
    return x


class SelectPolicy(nn.Module):
    def __init__(self, hidden_size, output_num):
        super(SelectPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.affine = nn.Linear(hidden_size, output_num)

    def forward(self, inputs):
        out = self.affine(inputs)
        out = F.sigmoid(out)
        return out


# changed reward

class OnlyAttentionFixErrorStructedPresentation(nn.Module):

    def __init__(self, action_num, vocabulary_size, hidden_size, encoder_num, num_heads, dropout_p=0.1, normalize_type=None):
        super(OnlyAttentionFixErrorStructedPresentation, self).__init__()
        self.action_num = action_num
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.encoder_num = encoder_num
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.normalize_type = normalize_type

        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.position_encode_fn = add_position_encode
        self.dropout = nn.Dropout(dropout_p)

        self.encoder_list = nn.ModuleList()
        for i in range(encoder_num):
            self.encoder_list.add_module('encoder_{}' + str(i),
                                         SelfAttentionEncoder(hidden_size, dropout_p=dropout_p, num_heads=num_heads, normalize_type=normalize_type))
        self.out = nn.Linear(hidden_size, action_num)

    def forward(self, inputs, input_lengths):
        pass


def calculate_rewards():
    pass


def split_and_combine_code_by_stay_label():
    pass


def check_contain_all_errors():
    pass
