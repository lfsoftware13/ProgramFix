import os
import random

import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class GGNNLayer(nn.Module):
    def __init__(self, hidden_state_size):
        super().__init__()
        self.gru_cell = nn.GRUCell(hidden_state_size, hidden_state_size)
        self.b = nn.Parameter(torch.randn(1, 1, hidden_state_size))

    def forward(self, x, adj):
        """
        :param x: shape [batch_size, seq, dim]
        :param adj: [batch_size, seq, seq]
        :return:
        """
        a = torch.bmm(adj, x) + self.b
        batch_size, seq, dim = x.shape
        o = self.gru_cell(a.view(-1, dim), x.view(-1, dim))
        return o.view(batch_size, seq, dim)


class MultiIterationGraph(nn.Module):
    def __init__(self, graph_propagate, graph_itr):
        super().__init__()
        self.graph_propagate = graph_propagate
        self.graph_itr = graph_itr

    def forward(self, x, adj):
        for _ in range(self.graph_itr):
            x = x + self.graph_propagate(x, adj)
        return x

