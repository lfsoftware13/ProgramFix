from abc import ABC
from typing import NamedTuple, Sequence, Iterable

import numpy as np

from common.problem_util import to_cuda
from model.sensibility_baseline.rnn_pytorch import SensibilityBiRnnModel
from model.sensibility_baseline.utility_class import Vind

import torch.nn.functional as F

import torch

class TokenResult(NamedTuple):
    forwards: np.ndarray
    backwards: np.ndarray


class DualLSTMModel(ABC):
    """
    A wrapper for accessing an individual Keras-defined model, for prediction
    only!
    """

    def predict_file(self, vector: Sequence[Vind]) -> Iterable[TokenResult]:
        """
        Produces prediction results for each token in the file.

        A stream of of (forwards, backwards) tuples, one for each
        cooresponding to a token in the file.

        forwards and backwards are each arrays of floats, having the size of
        the vocabulary. Each element is the probability of that vocabulary
        entry index being in the source file at the particular location.
        """


class DualLSTMModelWrapper(DualLSTMModel):
    def __init__(self,
                 rnn_model: SensibilityBiRnnModel):
        self.rnn_model = rnn_model

    def predict_file(self, vector: Sequence[Vind]) -> Iterable[TokenResult]:
        seq = to_cuda(torch.LongTensor([vector]))
        length = to_cuda(torch.LongTensor([len(vector)]))
        forward, backward = self.rnn_model(seq, length)
        forward = F.softmax(forward, dim=-1)
        backward = F.softmax(backward, dim=-1)
        return [TokenResult(forward[0, s, :].numpy(), backward[0, s, :].numpy()) for s in range(forward.size()[1])]
