import torch
import torch.nn as nn
import torch.nn.functional as F

from common import torch_util
from common.graph_embedding import GGNNLayer
from common.torch_util import create_sequence_length_mask
from seq2seq.models import EncoderRNN


class GraphEncoder(nn.Module):
    def __init__(self,
                 hidden_size=300,
                 graph_embedding='ggnn',
                 graph_parameter={},
                 pointer_type='itr',
                 embedding=None, embedding_size=400,
                 p2_type='static',
                 p2_step_length=0,
                 do_embedding=True
                 ):
        """
        :param hidden_size: The hidden state size in the model
        :param graph_embedding: The graph propagate method, option:["ggnn", "graph_attention", "rnn"]
        :param pointer_type: The method to point out the begin and the end, option:["itr", "query"]
        """
        super().__init__()
        self.pointer_type = pointer_type
        self.embedding = embedding
        self.p2_type = p2_type
        self.p2_step_length = p2_step_length
        self.do_embedding = do_embedding
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
            self.graph = MultiIterationGraphWrapper(hidden_size=hidden_size, **graph_parameter)
        elif graph_embedding == 'rnn':
            self.graph = RNNGraphWrapper(hidden_size=hidden_size, parameter=graph_parameter)
        elif graph_embedding == 'mixed':
            self.graph = MixedRNNGraphWrapper(hidden_size, **graph_parameter)

    def forward(self,
                adjacent_matrix,
                input_seq,
                copy_length,
                p1_target=None,
                ):
        if self.do_embedding:
            input_seq = self.embedding(input_seq)
            input_seq = self.size_transform(input_seq)
        input_seq = self.graph(input_seq, adjacent_matrix, copy_length)
        # batch_size = input_seq.shape[0]
        # if self.pointer_type == 'itr':
        #     p1_pointer_mask = torch_util.create_sequence_length_mask(copy_length-1, max_len=input_seq.shape[1])
        #     p2_pointer_mask = torch_util.create_sequence_length_mask(copy_length, max_len=input_seq.shape[1])
        #     input_seq0 = input_seq
        #     input_seq1 = self.graph(input_seq, adjacent_matrix, copy_length)
        #     input_seq2 = self.graph(input_seq1, adjacent_matrix, copy_length)
        #     p1 = self.pointer_transform(torch.cat((input_seq0, input_seq1), dim=-1), ).squeeze(-1)
        #     p1.data.masked_fill_(~p1_pointer_mask, -float("inf"))
        #     p2 = self.pointer_transform(torch.cat((input_seq0, input_seq2), dim=-1), ).squeeze(-1)
        #     p2.data.masked_fill_(~p2_pointer_mask, -float("inf"))
        #     input_seq = self.graph(input_seq2, adjacent_matrix, copy_length)
        # elif self.pointer_type == 'query':
        #     p1_pointer_mask = torch_util.create_sequence_length_mask(copy_length-1, max_len=input_seq.shape[1])
        #     p2_pointer_mask = torch_util.create_sequence_length_mask(copy_length, max_len=input_seq.shape[1])
        #     p1 = self.p1_pointer_network(input_seq, query=self.query_tensor, mask=p1_pointer_mask)
        #     p1_state = torch.sum(F.softmax(p1, dim=-1).unsqueeze(-1) * input_seq, dim=1)
        #     p2_query = self.up_data_cell(p1_state, self.query_tensor.unsqueeze(0).expand(batch_size, -1))
        #     if self.p2_type == 'static':
        #         p2 = self.p2_pointer_network(input_seq, query=torch.unsqueeze(p2_query, dim=1), mask=p2_pointer_mask)
        #     elif self.p2_type == 'step':
        #         p2_seq = torch.unbind(input_seq)
        #         if p1_target is None:
        #             p1_target = torch.argmax(p1, dim=1)
        #         p2_seq = [seq[p1_t+1:p1_t+1+self.p2_step_length] for seq, p1_t in zip(p2_seq, p1_target)]
        #         p2_seq = [torch.cat((seq, torch.zeros(self.p2_step_length-len(seq), input_seq.shape[-1]).to(input_seq.device))) for seq in p2_seq]
        #         p2_seq = [seq.unsqueeze(0) for seq in p2_seq]
        #         p2_seq = torch.cat(p2_seq, dim=0)
        #         p2 = self.p2_pointer_network(p2_seq, query=torch.unsqueeze(p2_query, dim=1))
        #     elif self.p2_type == 'sequence':
        #         p1 = torch.ones(p1.shape).float().to(p1.device)
        #         p1_sequence_mask = torch.zeros(p1.shape).byte().to(p1.device)
        #         p1_sequence_mask[:, 0] = 1
        #         p1.data.masked_fill_(~p1_sequence_mask, -float('inf'))
        #         p2 = torch.ones(p1.shape).float().to(p1.device)
        #         p2_sequence_mask = torch.zeros(p1.shape).byte().to(p1.device)
        #         for i, c in enumerate(copy_length.tolist()):
        #             p2_sequence_mask[i, c-1] = 1
        #         p2.data.masked_fill_(~p2_sequence_mask, -float('inf'))
        #     else:
        #         raise ValueError("No p2_type is:{}".format(self.p2_type))
        # else:
        #     raise ValueError("No point type is:{}".format(self.pointer_type))
        return None, None, input_seq


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


class MultiIterationGraphWrapper(nn.Module):
    def __init__(self,
                 hidden_size,
                 graph_type,
                 graph_itr,
                 dropout_p=0,
                 ):
        super().__init__()
        self.graph_itr = graph_itr
        self.dropout = nn.Dropout(dropout_p)
        self.inner_graph_itr = 1
        if graph_type == 'ggnn':
            self.graph = GGNNLayer(hidden_size)

    def forward(self, x, adj, copy_length):
        for i in range(self.graph_itr):
            x = x + self.graph(x, adj)
            if i < self.graph_itr - 1:
                x = self.dropout(x)
        return x


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
