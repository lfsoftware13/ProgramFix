import torch
from torch.distributions import Categorical
import torch.nn as nn


class AModel(nn.Module):
    def __init__(self, hidden_size):
        super(AModel, self).__init__()
        self.hidden_size = hidden_size
        self.affine = nn.Linear(hidden_size, hidden_size)
        self.binary_out = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        out = self.affine(inputs)
        binary = self.binary_out(out)
        sig_value = torch.sigmoid(binary)
        label = sig_value > 0.5
        return out

class BModel(nn.Module):
    def __init__(self, hidden_size):
        super(BModel, self).__init__()
        self.hidden_size = hidden_size
        self.affine = nn.Linear(hidden_size, hidden_size)
        self.base = AModel(hidden_size)

    def forward(self, inputs):
        out = self.affine(inputs)
        out_a = self.base(out)
        return out_a


def need_batch_first():
    batch_size = 4
    max_length = 3
    hidden_size = 2
    n_layers = 1
    feature_dim = 1

    # container
    batch_in = torch.zeros((batch_size, max_length, feature_dim))

    # data
    vec_1 = torch.FloatTensor([[1, 2, 3]])
    vec_2 = torch.FloatTensor([[1, 2, 0]])
    vec_3 = torch.FloatTensor([[1, 0, 0]])
    vec_4 = torch.FloatTensor([[2, 0, 0]])

    batch_in[0] = torch.transpose(vec_1, dim0=0, dim1=1)
    batch_in[1] = torch.transpose(vec_2, dim0=0, dim1=1)
    batch_in[2] = torch.transpose(vec_3, dim0=0, dim1=1)
    batch_in[3] = torch.transpose(vec_4, dim0=0, dim1=1)

    batch_in = torch.autograd.Variable(batch_in)
    print(batch_in.size())
    seq_lengths = [3, 2, 1, 1]  # list of integers holding information about the batch size at each sequence step

    # pack it
    pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
    print(pack)

    rnn = nn.RNN(feature_dim, hidden_size, n_layers)
    h0 = torch.autograd.Variable(torch.zeros(n_layers, batch_size, hidden_size))

    # forward
    out, _ = rnn(pack, h0)

    # unpack
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    print(unpacked)
    print(unpacked.shape)


def model_cuda_test():
    m = AModel(2)
    a = [[2, 2],
         [2, 2]]
    res = m.forward(torch.Tensor(a))


def test_fn(x, y, z):
    return x+y+z


if __name__ == '__main__':
    # loss_function = nn.CrossEntropyLoss()
    # model = BModel(2)
    # a = [[2, 2],
    #      [2, 2]]
    # res = model.forward(torch.Tensor(a))
    # print(res)
    # loss = loss_function(res, torch.LongTensor([0, 0]))
    # loss.backward()
    # model.zero_grad()
    # print(model)

    # need_batch_first()
    # model_cuda_test()
    v = {'x': 1, 'y': 2, 'z': 3}
    res = test_fn(v)
    print(res)

    # a = [3, 6]
    # c = Categorical(torch.Tensor(a))
    # cou = [0 for i in range(len(a))]
    # for i in range(10000):
    #     k = c.sample()
    #     cou[c.sample().item()] += 1
    # print(cou)
