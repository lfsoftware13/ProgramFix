import operator
from collections import OrderedDict
from itertools import islice

import torch.nn.functional as F
from torch import nn, nn as nn
import torch
import typing
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence

from common.problem_util import to_cuda
from common.util import transform_id_to_token


def save_model(model: torch.nn.Module, path):
    print('save model: {}'.format(path))
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path, map_location={}):
    model.load_state_dict(torch.load(path, map_location=map_location))

def mask_softmax(logit, mask):
    logit = logit * mask
    logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
    logit = logit - logit_max
    logit_exp = torch.exp(logit) * mask
    softmax = logit_exp/torch.sum(logit_exp, dim=-1, keepdim=True)
    return softmax


def to_sparse(x, cuda=True, gpu_index=0):
    """ converts dense tensor x to sparse format """
    print(torch.typename(x))
    x_typename = torch.typename(x).split('.')[-1]
    if cuda:
        sparse_tensortype = getattr(torch.cuda.sparse, x_typename)
    else:
        sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    if cuda:
        return sparse_tensortype(indices, values, x.size(), device=torch.device('cuda:{}'.format(gpu_index)))
    else:
        return sparse_tensortype(indices, values, x.size())


def pack_padded_sequence(padded_sequence, length, batch_firse=False,GPU_INDEX=0):
    _, idx_sort = torch.sort(length, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    length = torch.index_select(length, 0, idx_sort)
    if padded_sequence.is_cuda:
        padded_sequence = torch.index_select(padded_sequence, 0, idx_sort.cuda(GPU_INDEX))
    else:
        padded_sequence = torch.index_select(padded_sequence, 0, idx_sort)
    return torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, list(length), batch_first=batch_firse), idx_unsort


def pad_packed_sequence(packed_sequence, idx_unsort, pad_value, batch_firse=False, GPU_INDEX=0):
    padded_sequence, length = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=batch_firse,
                                                                padding_value=pad_value)
    if padded_sequence.is_cuda:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort).cuda(GPU_INDEX)), length
    else:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort)), length


def pack_sequence(sequences, GPU_INDEX=0):
    length = torch.Tensor([len(seq) for seq in sequences])
    _, idx_sort = torch.sort(length, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)
    packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
    return packed_sequences, idx_unsort


def create_ori_index_to_packed_index_dict(batch_sizes):
    begin_index = 0
    end_index = 0
    res = {}
    for i in range(len(batch_sizes)):
        end_index += batch_sizes[i]
        for j in range(end_index-begin_index):
            res[(j, i)] = begin_index + j
        begin_index += batch_sizes[i]
    return res


def create_stable_log_fn(epsilon):
    def stable_log(softmax_value):
        softmax_value = torch.clamp(softmax_value, epsilon, 1.0-epsilon)
        return torch.log(softmax_value)
    return stable_log


def padded_tensor_one_dim_to_length(one_tensor, dim, padded_length, is_cuda=False, gpu_index=0, fill_value=0):
    before_encoder_shape = list(one_tensor.shape)
    before_encoder_shape[dim] = padded_length - before_encoder_shape[dim]
    expend_tensor = (torch.ones(before_encoder_shape) * fill_value)
    if is_cuda:
        expend_tensor = expend_tensor.cuda(gpu_index)
    padded_outputs = torch.cat((one_tensor, expend_tensor), dim=dim)
    return padded_outputs


class MultiRNNCell(RNNCellBase):
    def __init__(self, cell_list: typing.List[RNNCellBase]):
        super().__init__()
        for idx, module in enumerate(cell_list):
            self.add_module(str(idx), module)

    def reset_parameters(self):
        for cell in self._modules.values():
            cell.reset_parameters()

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MultiRNNCell(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, h_i, h_s):
        res_h = []
        for h, cell in zip(h_s, self._modules.values()):
            h = cell(h_i, h)
            res_h.append(h)
            if isinstance(cell, torch.nn.LSTMCell):
                h_i = h[0]
            else:
                h_i = h
        return h_i, res_h


def calculate_accuracy_of_code_completion(log_probs, target, ignore_token=None, topk_range=(1, 15), gpu_index=None):
    """
    compare the log probility of all possible token with target token. calculate the accuracy of the code.
    ensure dim[1] of log_probs(seq len) is the same as dim[1] of target.
    :param log_probs:
    :param target:
    :param ignore_token:
    :param save_name:
    :param topk_range: (min_k, max_k)
    :return:
    """
    # log_probs_size = [batch_size, seq_len, vocab]
    if isinstance(target, list):
        target = torch.LongTensor(target)
        if gpu_index is not None:
            target = target.cuda(gpu_index)
    if isinstance(log_probs, PackedSequence):
        log_probs = log_probs.data
    if isinstance(target, PackedSequence):
        target = target.data

    batch_size = log_probs.shape[0]
    vocab_size = log_probs.shape[-1]

    log_probs = log_probs.view(-1, vocab_size)
    target = target.view(-1)

    if log_probs.shape[0] != target.shape[0]:
        print('different shape between log_probs and target. log_probs: {}, target: {}'.format(log_probs.shape, target.shape))
        raise Exception('different shape between log_probs and target. log_probs: {}, target: {}'.format(log_probs.shape, target.shape))

    # if len(log_probs.shape) == 2:
    #     log_probs = log_probs.unsqueeze(dim=1)

    max_topk = max(*topk_range)
    min_topk = min(*topk_range)
    if min_topk < 1:
        min_topk = 1
    if max_topk < 1:
        max_topk = 1

    # top_k_ids_size = [batch_size, seq_len, max_topk]
    top_k_ids = torch.topk(log_probs, dim=1, k=max_topk)[1]

    # resize target to the same shape of top k ids
    target = torch.unsqueeze(target, dim=1)
    repeat_shape = [1] * len(target.shape)
    repeat_shape[-1] = max_topk
    repeat_target = target.repeat(*repeat_shape)
    equal_list = torch.eq(top_k_ids, repeat_target)

    if ignore_token is not None:
        mask = torch.ne(target, ignore_token)
        zero_tensor = torch.zeros(equal_list.shape).byte()
        if gpu_index is not None:
            zero_tensor = zero_tensor.cuda(gpu_index)
        equal_list = torch.where(mask, equal_list, zero_tensor)

    result = {}
    for k in range(min_topk, max_topk+1):
        result[k] = equal_list[:, min_topk-1:k].sum().item()
    return result


def get_predict_and_target_tokens(log_probs, target, id_to_word_fn, k=1, offset=0):
    dim_len = len(log_probs.shape)
    softmaxed_probs = F.softmax(log_probs, dim=dim_len - 1)
    top_k_probs, top_k_ids = torch.topk(softmaxed_probs, dim=2, k=k)
    top_k_ids = top_k_ids.tolist()
    batch_predict = []
    batch_target = []
    for i, one in enumerate(top_k_ids):
        # one shape = [seq_len, k]
        predict_tokens = [transform_id_to_token(one_position, id_to_word_fn, offset=offset) for one_position in one]
        out_token = transform_id_to_token(target[i], id_to_word_fn, offset=offset)
        batch_predict += [predict_tokens]
        batch_target += [out_token]
    return batch_predict, batch_target, top_k_probs.tolist()


def spilt_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).

    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer

    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    # reshape last dim
    x_shape = list(x.shape)
    # x_shape += x_shape[-1]//num_heads
    x_shape = x_shape[:-1] + [num_heads, x_shape[-1]//num_heads]
    x = x.view(x_shape)

    x = torch.transpose(x, dim0=-3, dim1=-2)
    return x


def create_sequence_length_mask(token_length, max_len=None, gpu_index=None):
    if max_len is None:
        max_len = torch.max(token_length).item()
    idxes = torch.arange(0, max_len, out=torch.Tensor(max_len)).unsqueeze(0)  # some day, you'll be able to directly do this on cuda
    idxes = idxes.to(token_length.device)
    # mask = autograd.Variable((trans_to_cuda(idxes) < token_length.unsqueeze(1)).float())
    mask = (idxes < token_length.unsqueeze(1).float())
    return mask


def permute_last_dim_to_second(log_probs):
    permute_shape = [i for i in range(len(log_probs.shape))]
    permute_shape.insert(1, permute_shape[-1])
    permute_shape = permute_shape[:-1]
    log_probs = log_probs.permute(*permute_shape)
    return log_probs


def expand_tensor_sequence_len(t, max_len, fill_value=0, dim=1):
    t_len = t.shape[dim]
    if max_len == t_len:
        return t
    expand_shape = list(t.shape)
    expand_shape[dim] = 1
    one_t = to_cuda(torch.ones(*expand_shape).float()) * fill_value
    expand_t = one_t.expand(*[-1 for i in range(dim)], max_len - t_len, *[-1 for i in range(len(t.shape) - 1 - dim)])
    if t.data.type() == 'torch.cuda.LongTensor' or t.data.type() == 'torch.LongTensor':
        expand_t = expand_t.long()
    elif t.data.type() == 'torch.cuda.ByteTensor' or t.data.type() == 'torch.ByteTensor':
        expand_t = expand_t.byte()
    res_t = torch.cat([t, expand_t], dim=dim)
    return res_t


def expand_tensor_sequence_to_same(t1, t2, fill_value=0):
    len1 = t1.shape[1]
    len2 = t2.shape[1]
    expand_len = max(len1, len2)
    t1 = expand_tensor_sequence_len(t1, expand_len, fill_value)
    t2 = expand_tensor_sequence_len(t2, expand_len, fill_value)
    return t1, t2


def expand_tensor_sequence_list_to_same(t_list, fill_value=0, dim=1):
    max_len = t_list[0].shape[dim]
    for t in t_list[1:]:
        if t.shape[dim]>max_len:
            max_len = t.shape[dim]

    expand_t_list = [expand_tensor_sequence_len(t, max_len=max_len, fill_value=fill_value, dim=dim) for t in t_list]
    return expand_t_list


def expand_output_and_target_sequence_len(model_output, model_target, fill_value=0):
    res = [expand_tensor_sequence_to_same(out, tar, fill_value=fill_value) for out, tar in zip(model_output, model_target)]
    return list(zip(*res))


class DynamicDecoder(object):
    def __init__(self, start_label, end_label, pad_label, decoder_fn, create_next_output_fn, max_length):
        self.start_label = start_label
        self.end_label = end_label
        self.pad_label = pad_label
        self.decoder_fn = decoder_fn
        self.create_next_output_fn = create_next_output_fn
        self.max_length = max_length

    def decoder(self, encoder_output, endocer_hidden, encoder_mask, **kwargs):
        batch_size = encoder_output.shape[0]
        continue_mask = to_cuda(torch.ByteTensor([1 for i in range(batch_size)]))
        outputs = to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)]))
        decoder_output_list = []
        outputs_list = []
        hidden = endocer_hidden
        error_list = [0 for i in range(batch_size)]
        
        for i in range(self.max_length):
            one_step_decoder_output, hidden, error_ids = self.decoder_fn(outputs, continue_mask, start_index=i, hidden=hidden,
                                                              encoder_output=encoder_output,
                                                              encoder_mask=encoder_mask,
                                                              **kwargs)
            if (error_ids is not None) and len(error_ids) != 0:
                error_ids_list = [0 for i in range(batch_size)]
                for err in error_ids:
                    # print('error index: {}'.format(err))
                    error_list[err] = 1
                    error_ids_list[err] = 1
                error_ids_tensor = to_cuda(torch.ByteTensor(error_ids_list))
                continue_mask = continue_mask & ~error_ids_tensor
                    # continue_mask[err] = 0
            decoder_output_list += [one_step_decoder_output]

            outputs = self.create_next_output_fn(one_step_decoder_output, **kwargs)
            outputs_list += [outputs]
            step_continue = torch.ne(outputs, self.end_label).view(batch_size)
            continue_mask = continue_mask & step_continue

            # try:
            if torch.sum(continue_mask) == 0:
                break
            # except Exception as e:
            #     print(e)
            #     print(error_list)
            #     print(outputs)
            #     print(step_continue)
            #     print(continue_mask)
            #     raise Exception(e)
        return decoder_output_list, outputs_list, error_list


class BeamSearchDynamicDecoder(object):
    def __init__(self, start_label, end_label, pad_label, decoder_fn, create_beam_next_output_fn, max_length, beam_size=5):
        self.start_label = start_label
        self.end_label = end_label
        self.pad_label = pad_label
        self.decoder_fn = decoder_fn
        self.create_beam_next_output_fn = create_beam_next_output_fn
        self.max_length = max_length
        self.beam_size = beam_size

    def decoder(self, encoder_output, endocer_hidden, encoder_mask, **kwargs):
        batch_size = encoder_output.shape[0]
        continue_mask_stack = to_cuda(torch.ByteTensor([[1 for _ in range(self.beam_size)] for i in range(batch_size)]))
        beam_outputs = to_cuda(torch.LongTensor([[[self.start_label] for _ in range(self.beam_size)] for i in range(batch_size)]))
        # outputs_stack = [to_cuda(torch.LongTensor([[self.start_label] for i in range(batch_size)])) for _ in range(self.beam_size)]
        probability_stack = to_cuda(torch.FloatTensor([[0.0 for _ in range(self.beam_size)] for _ in range(batch_size)]))
        decoder_output_list = []
        outputs_list = []
        hidden_stack = [endocer_hidden for _ in range(self.beam_size)]
        error_stack = to_cuda(torch.ByteTensor([[0 for _ in range(self.beam_size)] for i in range(batch_size)]))

        for i in range(self.max_length):
            # beam * (output * [batch, 1, ...])
            # beam_one_step_decoder_output = []
            beam_outputs_list = []
            beam_log_probs_list = []
            beam_decoder_output_list = []
            # beam * [batch, hidden]
            beam_hidden_list = []
            # beam * [batch, error_count]
            beam_error_ids = []
            for b in range(self.beam_size):
                outputs = beam_outputs[:, b]
                continue_mask = continue_mask_stack[:, b]
                hidden = hidden_stack[b]
                one_step_decoder_output, hidden, error_ids = self.decoder_fn(outputs, continue_mask, start_index=i,
                                                                             hidden=hidden,
                                                                             encoder_output=encoder_output,
                                                                             encoder_mask=encoder_mask,
                                                                             **kwargs)

                # if (error_ids is not None) and len(error_ids) != 0:
                error_ids_list = [0 for i in range(batch_size)]
                for err in error_ids:
                    error_ids_list[err] = 1
                error_ids_tensor = to_cuda(torch.ByteTensor(error_ids_list))
                beam_error_ids += [error_ids_tensor]

                beam_hidden_list += [hidden]

                # one_beam_outputs: [batch, beam, seq]
                # beam_probs: [batch, beam]
                # one_beam_decoder_output: tuple of [batch, beam, seq]
                one_beam_outputs, one_beam_log_probs, one_beam_decoder_output = self.create_beam_next_output_fn(one_step_decoder_output, continue_mask=continue_mask, beam_size=self.beam_size, **kwargs)
                beam_outputs_list += [one_beam_outputs]
                beam_log_probs_list += [one_beam_log_probs]
                beam_decoder_output_list += [one_beam_decoder_output]

                if i == 0:
                    break

            # beam_step_log_probs: [batch, outer_beam, inner_beam]
            beam_step_log_probs = torch.stack(beam_log_probs_list, dim=1)
            if i != 0:
                beam_total_log_probs = torch.unsqueeze(probability_stack, dim=-1) + torch.squeeze(beam_step_log_probs, dim=-1)
            else:
                beam_total_log_probs = torch.squeeze(beam_step_log_probs, dim=-1)
            probability_stack, sort_index = torch.topk(
                beam_total_log_probs.view(batch_size, beam_total_log_probs.shape[1] * beam_total_log_probs.shape[2]),
                k=self.beam_size, dim=-1)
            stack_sort_index = sort_index / self.beam_size

            # beam_outputs: [batch, outer_beam * inner_beam, seq]
            beam_outputs = beam_stack_and_reshape(beam_outputs_list)
            beam_outputs = batch_index_select(beam_outputs, sort_index, batch_size)
            outputs_list = [batch_index_select(outputs, stack_sort_index, batch_size)
                            for outputs in outputs_list]
            outputs_list += [beam_outputs]

            beam_decoder_output = [beam_stack_and_reshape(one_output_list)
                                   for one_output_list in zip(*beam_decoder_output_list)]
            beam_decoder_output = [batch_index_select(one_output, sort_index, batch_size)
                                   for one_output in beam_decoder_output]

            decoder_output_list = [[batch_index_select(one_output, stack_sort_index, batch_size)
                                    for one_output in decoder_output]
                                   for decoder_output in decoder_output_list]
            decoder_output_list += [beam_decoder_output]

            # beam_error = beam_stack_and_reshape(beam_error_ids)
            beam_error = torch.stack(beam_error_ids, dim=1)
            beam_error = batch_index_select(beam_error, stack_sort_index, batch_size=batch_size)
            beam_continue = torch.ne(beam_outputs, self.end_label).view(batch_size, self.beam_size)
            beam_continue = beam_continue & ~beam_error
            continue_mask_stack = batch_index_select(continue_mask_stack, stack_sort_index, batch_size) & beam_continue
            error_stack = batch_index_select(error_stack, stack_sort_index, batch_size) | beam_error

            if isinstance(beam_hidden_list[0], list):
                one_hidden_list = zip(*beam_hidden_list)
                hidden_stack = list(zip(*[deal_beam_hidden(one_hidden_beam_list, stack_sort_index, batch_size)
                                          for one_hidden_beam_list in one_hidden_list]))
            else:
                hidden_stack = deal_beam_hidden(beam_hidden_list, stack_sort_index, batch_size)

            # try:
            if torch.sum(continue_mask_stack) == 0:
                break
            # except Exception as e:
            #     print(e)
            #     print(error_list)
            #     print(outputs)
            #     print(step_continue)
            #     print(continue_mask)
            #     raise Exception(e)
        return decoder_output_list, outputs_list, error_stack


def deal_beam_hidden(hidden_list, stack_sort_index, batch_size):
    # hidden_shape = list(hidden_list[0].shape)
    hidden_list = [hidden.permute(1, 0, 2) for hidden in hidden_list]
    # beam_hidden = beam_stack_and_reshape(hidden_list)
    beam_hidden = torch.stack(hidden_list, dim=1)
    beam_hidden = batch_index_select(beam_hidden, stack_sort_index, batch_size)
    hidden_list = torch.unbind(beam_hidden, dim=1)
    hidden_list = [hidden.permute(1, 0, 2).contiguous() for hidden in hidden_list]
    return hidden_list


def beam_stack_and_reshape(beam_outputs_list):
    """

    :param beam_outputs_list: a list of tensor [batch, inner_beam, ...]. len(list): outer_beam
    :return: [batch, outer_beam * inner_beam, ...]
    """
    beam_outputs = torch.stack(beam_outputs_list, dim=1)
    beam_outputs_shape = beam_outputs.shape
    beam_outputs = beam_outputs.view(beam_outputs_shape[0], -1, *beam_outputs_shape[3:])
    return beam_outputs


def batch_index_select(beam_outputs, sort_index, batch_size):
    """

    :param beam_outputs_list: Tensor [batch, outer_beam * inner_beam, ...].
    :param sort_index: [batch, beam]. top beam-th index of log probs tensor [batch, beam * beam]
    :return:
    """
    tensor_list = []
    # beam_outputs: [batch, outer_beam * inner_beam, seq]
    for b_idx in range(batch_size):
        tmp_tensor = torch.index_select(beam_outputs[b_idx], dim=0, index=sort_index[b_idx])
        tensor_list += [tmp_tensor]
    # beam_output: [batch, beam, seq]
    beam_outputs = torch.stack(tensor_list, dim=0)
    return beam_outputs


def pad_last_dim_of_tensor_list(tensor_list, max_len=None, fill_value=0):
    total_len = [tensor.shape[-1] for tensor in tensor_list]
    if max_len is None:
        total_len = [tensor.shape[-1] for tensor in tensor_list]
        max_len = max(total_len)

    padded_tensor_list = [F.pad(tensor, (0, max_len-one_len), 'constant', fill_value) for tensor, one_len in zip(tensor_list, total_len)]
    return padded_tensor_list


class Update(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.update_value = nn.Linear(2*hidden_size, hidden_size)
        self.update_gate = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, original_value, to_update_value):
        x = torch.cat((original_value, to_update_value), dim=-1)
        update_value = F.tanh(self.update_value(x))
        update_gate = F.sigmoid(self.update_gate(x))
        return original_value*(1-update_gate) + update_value*update_gate


class MaskOutput(nn.Module):
    def __init__(self,
                 hidden_state_size,
                 vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_state_size)

    def forward(self, input_seq, grammar_index, grammar_mask, ):
        weight = self.embedding(grammar_index).permute(0, 2, 1)
        input_seq = input_seq.unsqueeze(1)
        o = torch.bmm(input_seq, weight).squeeze(1)
        o.data.masked_fill_(~grammar_mask, -float('inf'))
        return o


class SequenceMaskOutput(nn.Module):
    def __init__(self,
                 hidden_state_size,
                 vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_state_size)

    def forward(self, input_seq, grammar_index, grammar_mask, ):
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        hidden_size = input_seq.shape[2]
        weight = self.embedding(grammar_index).permute(0, 1, 3, 2)
        weight_hidden_size = weight.shape[-2]
        input_seq = input_seq.view(batch_size*seq_len, 1, hidden_size)
        weight = weight.contiguous().view(batch_size*seq_len, weight_hidden_size, -1)
        o = torch.bmm(input_seq, weight).view(batch_size, seq_len, -1)
        o.data.masked_fill_(~grammar_mask, -float('inf'))
        input_seq_mask = torch.ne(torch.sum(grammar_mask, dim=-1), 0)
        o.data.masked_fill_(~torch.unsqueeze(input_seq_mask, dim=-1), 0)
        return o


def stable_log(input, e=10e-5):
    stable_input = torch.clamp(input, e, 1 - e)
    output = torch.log(stable_input)
    return output


def remove_last_item_in_sequence(x, x_length, k=1):
    """

    :param x: [batch, seq, ...]
    :param l: [batch]
    :param k:
    :return:
    """
    x_list = torch.unbind(x, dim=0)
    remain_x_list = [torch.cat([one[:l - k], one[l:]], dim=0) for one, l in zip(x_list, x_length)]
    o = torch.stack(remain_x_list, dim=0)
    return o


def reverse_tensor(x, x_length):
    x_list = torch.unbind(x, dim=0)
    reverse_list = []
    for one, l in zip(x_list, x_length):
        idx = to_cuda(torch.arange(l.item()-1, -1, -1).long())
        r_one = one.index_select(dim=0, index=idx)
        reverse_list += [torch.cat([r_one, one[l:]], dim=0)]
    o = torch.stack(reverse_list, dim=0)
    return o


if __name__ == '__main__':
    a = [[0, 1, 2, 3, 4, 5],
         [0, 1, 2, 3, 4, 5],
         [0, 1, 2, 3, 4, 5],
         [0, 1, 2, 3, 4, 5]]
    a_t = torch.Tensor(a)
    l = torch.LongTensor([2, 3, 4, 6])
    r_a = remove_last_item_in_sequence(a_t, l, k=1)
    print(r_a)