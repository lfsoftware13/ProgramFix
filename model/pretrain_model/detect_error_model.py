import torch
import torch.nn as nn

from common.torch_util import create_sequence_length_mask
from model.pretrain_model.graph_encoder_model import GraphEncoder
from model.pretrain_model.transformer.Attention import ScaledDotProductAttention
from model.pretrain_model.transformer.transformer_sublayers import MultiHeadAttention


class ErrorDetectorModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query',
                 p2_type='static', p2_step_length=0, check_error_task=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.graph_encoder = GraphEncoder(hidden_size=hidden_size,
                 graph_embedding=graph_embedding,
                 graph_parameter=graph_parameter,
                 pointer_type=pointer_type,
                 embedding=self.embedding, embedding_size=hidden_size,
                 p2_type=p2_type,
                 p2_step_length=p2_step_length,
                 do_embedding=True)
        self.check_error_task = check_error_task
        if check_error_task:
            self.output = nn.Linear(hidden_size, 1)
        else:
            self.attention = MultiHeadAttention(1, hidden_size, hidden_size, hidden_size, dropout=0.2)
            self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, adjacent_matrix, inp_seq, inp_seq_len):
        _, _, encoder_logit = self.graph_encoder(adjacent_matrix, inp_seq, inp_seq_len)
        if self.check_error_task:
            output_logit = self.output(encoder_logit).squeeze(-1)
        else:
            mask = create_sequence_length_mask(inp_seq_len).unsqueeze(dim=-2)
            encoder_logit, _ = self.attention(encoder_logit, encoder_logit, encoder_logit, mask)
            output_logit = self.output(encoder_logit)
        return [output_logit]


def create_loss_fn(ignore_id):
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def loss_fn(output_logit, target_replaced):
        mask = torch.ne(output_logit, ignore_id)
        replaced_loss = bce_loss(output_logit, target_replaced) * mask.float()
        replaced_loss = torch.sum(replaced_loss) / torch.sum(mask)
        return replaced_loss
    return loss_fn


if __name__ == '__main__':
    graph_parameter = {"rnn_parameter": {'vocab_size': 1000,
                                                   'max_len': 500, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 }
    m = ErrorDetectorModel(400, 1000, 'mixed', graph_parameter, pointer_type='query', p2_type='step', p2_step_length=2)
