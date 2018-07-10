import json
import random
import torch

import numpy as np
import more_itertools

from c_parser.pycparser.pycparser.ply.lex import LexToken
from common.analyse_include_util import extract_include, replace_include_with_blank
from common.util import PaddedList
from torch.nn import functional as F


# ------------------------- produce part token by part position list using tokens tensor --------------------------- #
def combine_spilt_tokens_batch_with_tensor(tokens_tensor_total, ac_tokens_list, stay_labels_list, token_map_list,
                                           gap_tensor, begin_tensor, end_tensor, gap_token, begin_token, end_token, gpu_index=None):
    """

    :param tokens_tensor_total: [batch, seq, hidden, ...], a tensor contain all error tokens with hidden state
    :param ac_tokens_list: a list of ac tokens. it should not be a tensor
    :param stay_labels_list: [batch, seq], a python list of python list of action.
    :param token_map_list: a map list from error token index to ac token index
    :param gap_tensor: gap tensor value. shape [hidden, ...]
    :param begin_tensor: shape [hidden, ...]
    :param end_tensor: shape [hidden, ...]
    :param gap_token: gap token id
    :param begin_tokens: python list of begin token ids.
    :param end_tokens: python list of end token ids.
    :return:
    """
    if isinstance(tokens_tensor_total, torch.Tensor):
        tokens_tensor_total = torch.unbind(tokens_tensor_total, dim=0)
    if isinstance(ac_tokens_list, torch.Tensor):
        ac_tokens_list = torch.unbind(ac_tokens_list, dim=0)
    split_tokens = [combine_spilt_tokens_with_tensor(tokens, ac_tokens, stay_labels, token_map, gap_token, gap_tensor)
                    for tokens, ac_tokens, stay_labels, token_map in
                    zip(tokens_tensor_total, ac_tokens_list, stay_labels_list, token_map_list)]
    part_tokens_list, part_ac_tokens_list = list(zip(*split_tokens))

    part_tokens_list = [add_begin_end_token(part_tokens, begin_token, begin_tensor, end_token, end_tensor) for part_tokens in part_tokens_list]
    part_ac_tokens_list = [add_begin_end_token(part_ac_tokens, begin_token, begin_tensor, end_token, end_tensor) for part_ac_tokens in part_ac_tokens_list]

    tokens_tensor = concat_sequence_tensor_to_one_batch_tensor(part_tokens_list, dim=0, gpu_index=gpu_index)
    token_length = [token.shape[0] for token in part_tokens_list]
    if isinstance(part_ac_tokens_list[0], torch.Tensor):
        ac_token_length = [token.shape[0] for token in part_ac_tokens_list]
        part_ac_tokens_list = concat_sequence_tensor_to_one_batch_tensor(part_ac_tokens_list, dim=0, gpu_index=gpu_index)
    else:
        ac_token_length = [len(token) for token in part_ac_tokens_list]

    return tokens_tensor, token_length, part_ac_tokens_list, ac_token_length


def concat_sequence_tensor_to_one_batch_tensor(input_list, dim=0, filled_value=0, gpu_index=None):
    sequence_len = [inp.shape[dim] for inp in input_list]
    max_len = max(sequence_len)
    one_fill_shape = list(input_list[0].shape)
    one_fill_shape[dim] = 1
    one_fill_tensor = torch.ones(one_fill_shape) * filled_value
    if gpu_index is not None:
        one_fill_tensor = one_fill_tensor.cuda(gpu_index)

    fill_expand_shape = [-1 for i in range(len(one_fill_shape))]
    fill_dim_len = [max_len - inp.shape[dim] for inp in input_list]
    for i in range(len(input_list)):
        fill_expand_shape[dim] = fill_dim_len[i]
        input_list[i] = torch.cat([input_list[i], one_fill_tensor.expand(*fill_expand_shape)], dim=dim)
    batch_tensor = torch.stack(input_list, dim=0)
    return batch_tensor


def add_begin_end_token(tokens, begin_token, begin_tensor, end_token, end_tensor):
    add_begin = begin_token is not None and begin_tensor is not None
    add_end = end_token is not None and end_tensor is not None
    if add_begin:
        if isinstance(tokens, torch.Tensor):
            tokens = torch.cat([torch.unsqueeze(begin_tensor, dim=0), tokens], dim=0)
        elif isinstance(tokens, list):
            tokens = [begin_token] + tokens
    if add_end:
        if isinstance(tokens, torch.Tensor):
            tokens = torch.cat([tokens, torch.unsqueeze(end_tensor, dim=0)], dim=0)
        elif isinstance(tokens, list):
            tokens = tokens + [end_token]
    return tokens


def combine_spilt_tokens_with_tensor(tokens, ac_tokens, stay_labels, token_map, gap_token, gap_tensor):
    position_list = produce_position_list_by_stay_label(stay_labels, len(token_map))
    split_position_list = create_split_position_by_position_list(position_list)
    split_tokens = [tokens[start:end+1] for start, end in split_position_list]
    part_tokens = None
    if isinstance(tokens, torch.Tensor):
        part_tokens = join_gap_in_token_tensor_list(split_tokens, gap_tensor)
    elif isinstance(tokens, list):
        part_tokens = join_gap_in_token_list(split_tokens, gap_token)

    part_ac_tokens = None
    if ac_tokens is not None:
        split_ac_position_list = [(token_map[start], token_map[end]) for start, end in split_position_list]
        split_ac_tokens = [ac_tokens[start:end+1] for start, end in split_ac_position_list]
        if isinstance(ac_tokens, torch.Tensor):
            part_ac_tokens = join_gap_in_token_tensor_list(split_ac_tokens, gap_tensor)
        elif isinstance(ac_tokens, list):
            part_ac_tokens = join_gap_in_token_list(split_ac_tokens, gap_token)
    return part_tokens, part_ac_tokens


def join_gap_in_token_tensor_list(split_token_tensors, gap_tensor):
    """

    :param split_token_tensors: a list of [part_len, hidden, ...]
    :param gap_tensor: [hidden, ...]. the same shape as split_token_tensors[0].shape[1:]
    :return:
    """
    gap_tensor = torch.unsqueeze(gap_tensor, dim=0)
    part_tokens = []
    part_tokens += [split_token_tensors[0]]
    for s in split_token_tensors[1:]:
        if gap_tensor is not None:
            part_tokens += [gap_tensor]
        part_tokens += [s]
    part_tokens_tensor = torch.cat(part_tokens, dim=0)
    return part_tokens_tensor


# ------------------------- produce part token by part position list using tokens list --------------------------- #
def combine_spilt_tokens_batch(tokens_list, ac_tokens_list, stay_labels_list, token_map_list, gap_token, begin_tokens, end_tokens):
    split_tokens = [combine_spilt_tokens(tokens, ac_tokens, stay_labels, token_map, gap_token)
                    for tokens, ac_tokens, stay_labels, token_map in zip(tokens_list, ac_tokens_list, stay_labels_list, token_map_list)]
    part_tokens_list, part_ac_tokens_list = list(zip(*split_tokens))
    if begin_tokens is not None:
        part_tokens_list = [begin_tokens + part_tokens for part_tokens in part_tokens_list]
        part_ac_tokens_list = [begin_tokens + part_ac_tokens for part_ac_tokens in part_ac_tokens_list]
    if end_tokens is not None:
        part_tokens_list = [part_tokens + end_tokens for part_tokens in part_tokens_list]
        part_ac_tokens_list = [part_ac_tokens + end_tokens for part_ac_tokens in part_ac_tokens_list]
    return part_tokens_list, part_ac_tokens_list


def combine_spilt_tokens(tokens, ac_tokens, stay_labels, token_map, gap_token):
    position_list = produce_position_list_by_stay_label(stay_labels)
    split_position_list = create_split_position_by_position_list(position_list)
    split_tokens = [tokens[start:end+1] for start, end in split_position_list]
    part_tokens = join_gap_in_token_list(split_tokens, gap_token)

    part_ac_tokens = None
    if ac_tokens is not None:
        split_ac_position_list = [(token_map[start], token_map[end]) for start, end in split_position_list]
        split_ac_tokens = [ac_tokens[start:end+1] for start, end in split_ac_position_list]
        part_ac_tokens = join_gap_in_token_list(split_ac_tokens, gap_token)
    return part_tokens, part_ac_tokens


def produce_position_list_by_stay_label(stay_labels, token_len):
    position_list = [i if stay_labels[i] else None for i in range(len(stay_labels))]
    position_list = list(filter(lambda x: x is not None, position_list))
    # tmp add special case to avoid empty position list
    if len(position_list) == 0:
        position_list += [token_len-1]
    return position_list


def create_split_position_by_position_list(position_list):
    is_first_fn = lambda ind: ind == 0 or (position_list[ind] != position_list[ind-1]+1)
    is_last_fn = lambda ind: ind == (len(position_list)-1) or (position_list[ind]+1) != position_list[ind+1]
    first_position_index = list(filter(is_first_fn, range(len(position_list))))
    last_position_index = list(filter(is_last_fn, range(len(position_list))))
    first_position = [position_list[i] for i in first_position_index]
    last_position = [position_list[i] for i in last_position_index]
    split_position = list(zip(first_position, last_position))
    return split_position


def join_gap_in_token_list(split_tokens, gap_token):
    part_tokens = []
    part_tokens += split_tokens[0]
    for s in split_tokens[1:]:
        if gap_token is not None:
            part_tokens += [gap_token]
        part_tokens += s
    return part_tokens


# -------------------------- random choose token for pre-train ------------------------------- #
def choose_token_random_batch(token_lengths, error_labels, random_value=0.2):
    error_masks = [[i > 0 for i in error_mask] if error_mask is not None else None for error_mask in error_labels]
    random_fn = lambda x: random.random() < x
    stay_labels = [[random_fn(random_value) for i in range(one_length)] for one_length in token_lengths]
    masked_stay_labels = [[lab or mas for lab, mas in zip(stay_label, error_mask)] if error_mask is not None else stay_label for stay_label, error_mask in zip(stay_labels, error_masks)]
    return masked_stay_labels


def check_action_include_all_error(action_tensor, error_labels):
    """
    check if actions have including all error related labels
    :param action_tensor: [batch, ...]
    :param error_labels: [batch, ...]. the same shape as action tensor.
    :return: ByteTensor with shape [batch,]. The position is 1(True) if action include all error label else 0.
    """
    error_but_not_choose = error_labels.byte() & (~action_tensor.byte())
    error_but_not_choose = error_but_not_choose.view(error_but_not_choose.shape[0], -1)
    result = (torch.sum(error_but_not_choose, dim=1) < 1)
    return result


def force_include_all_error_in_action(action_tensor, error_labels):
    combine_choose = action_tensor.byte() | error_labels.byte()
    return combine_choose


# ---------------------------- parse xy, create error tokens and ac tokens with a token map -------------------- #

CHANGE = 0
INSERT = 1
DELETE = 2

def parse_test_tokens(df, data_type, keyword_vocab, tokenize_fn):
    df['code_tokens'] = df['code'].map(tokenize_fn)
    df = df[df['code_tokens'].map(lambda x: x is not None)].copy()
    df['code_tokens'] = df['code_tokens'].map(list)
    print('after tokenize: ', len(df.index))

    df['code_words'] = df['code_tokens'].map(create_name_list_by_LexToken)
    transform_word_to_id_fn = lambda name_list: keyword_vocab.parse_text([name_list], False)[0]
    df['error_code_word_id'] = df['code_words'].map(transform_word_to_id_fn)
    return df['error_code_word_id']


def parse_error_tokens_and_action_map(df, data_type, keyword_vocab, sort_fn=None, tokenize_fn=None):
    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(tokenize_fn)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()
    df['ac_code_obj'] = df['ac_code_obj'].map(list)
    print('after tokenize: ', len(df.index))

    df = df.apply(create_error_list, axis=1, raw=True, sort_fn=sort_fn)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create error: ', len(df.index))

    df = df.apply(create_token_id_input, axis=1, raw=True, keyword_voc=keyword_vocab)
    df = df[df['res'].map(lambda x: x is not None)].copy()
    print('after create token id: ', len(df.index))

    df['ac_code_word'] = df['ac_code_obj'].map(create_name_list_by_LexToken)
    transform_word_to_id_fn = lambda name_list: keyword_vocab.parse_text([name_list], False)[0]
    df['ac_code_word_id'] = df['ac_code_word'].map(transform_word_to_id_fn)

    get_first_fn = lambda x: x[0]
    df['error_code_word_id'] = df['token_id_list'].map(get_first_fn)

    create_token_map_by_action_fn = lambda one: create_token_map_by_action(one['error_code_word_id'], one['action_list'])
    df['token_map'] = df.apply(create_token_map_by_action_fn, raw=True, axis=1)

    create_tokens_error_mask_fn = lambda one: create_tokens_error_mask(one['error_code_word_id'], one['action_list'])
    df['error_mask'] = df.apply(create_tokens_error_mask_fn, raw=True, axis=1)
    df = df[df['error_mask'].map(lambda x: x is not None)].copy()
    print('after error_mask id: ', len(df.index))

    df['is_copy'] = df.apply(create_is_copy_for_ac_tokens, raw=True, axis=1)

    return df['error_code_word_id'], df['ac_code_word_id'], df['token_map'], df['error_mask'], df['is_copy']


def create_token_id_input(one, keyword_voc):
    token_name_list = one['token_name_list']
    token_id_list = []
    len_list = []
    for name_list in token_name_list:
        id_list = keyword_voc.parse_text([name_list], False)[0]
        if id_list == None:
            one['res'] = None
            return one
        len_list.append(len(id_list))
        token_id_list.append(id_list)
    one['token_id_list'] = token_id_list
    one['token_length_list'] = len_list
    return one



def create_error_list(one, sort_fn=None):
    import json
    ac_code_obj = one['ac_code_obj']
    action_token_list = json.loads(one['action_character_list'])
    if sort_fn is not None:
        action_token_list = sort_fn(action_token_list)

    def cal_token_pos_bias(action_list, cur_action):
        bias = 0
        cur_token_pos = cur_action['token_pos']
        for act in action_list:
            if act['act_type'] == INSERT and (cur_action['act_type'] == DELETE or cur_action['act_type'] == CHANGE) and cur_token_pos >= act['token_pos']:
                bias += 1
            elif act['act_type'] == INSERT and cur_token_pos > act['token_pos']:
                bias += 1
            elif act['act_type'] == DELETE and cur_token_pos > act['token_pos']:
                bias -= 1
        return bias

    token_pos_list = [act['token_pos'] if act['act_type'] != INSERT else -1 for act in action_token_list]
    token_pos_list = list(filter(lambda x: x != -1, token_pos_list))
    has_repeat_action_fn = lambda x: len(set(x)) < len(x)
    if has_repeat_action_fn(token_pos_list):
        one['res'] = None
        return one

    token_bias_list = [cal_token_pos_bias(action_token_list[0:i], action_token_list[i]) for i in range(len(action_token_list))]

    token_name_list = []
    action_list = []
    for act, token_bias in zip(action_token_list, token_bias_list):
        # ac_pos = act['ac_pos']
        ac_type = act['act_type']
        ac_token_pos = act['token_pos']
        real_token_pos = ac_token_pos + token_bias

        if ac_type == INSERT:
            to_char = act['to_char']
            tok = LexToken()
            tok.value = to_char
            tok.lineno = -1
            tok.type = ""
            tok.lexpos = -1
            ac_code_obj = ac_code_obj[0:real_token_pos] + [tok] + ac_code_obj[real_token_pos:]
            action = {'type': DELETE, 'pos': real_token_pos * 2 + 1, 'token': to_char}
            name_list = create_name_list_by_LexToken(ac_code_obj)
            token_name_list = [name_list] + token_name_list
            action_list = [action] + action_list
        elif ac_type == DELETE:
            from_char = act['from_char']
            ac_code_obj = ac_code_obj[0: real_token_pos] + ac_code_obj[real_token_pos+1:]
            action = {'type': INSERT, 'pos': real_token_pos * 2, 'token': from_char}
            name_list = create_name_list_by_LexToken(ac_code_obj)
            token_name_list = [name_list] + token_name_list
            action_list = [action] + action_list
        elif ac_type == CHANGE:
            from_char = act['from_char']
            to_char = act['to_char']
            tok = LexToken()
            tok.value = to_char
            tok.lineno = -1
            tok.type = ""
            tok.lexpos = -1
            action = {'type': CHANGE, 'pos': real_token_pos * 2 + 1, 'token': from_char}
            ac_code_obj = ac_code_obj[0: real_token_pos] + [tok] +ac_code_obj[real_token_pos + 1:]
            name_list = create_name_list_by_LexToken(ac_code_obj)
            token_name_list = [name_list] + token_name_list
            action_list = [action] + action_list

    if len(action_list) == 0:
        one['res'] = None
        return one
    one['token_name_list'] = token_name_list
    one['action_list'] = action_list
    one['copy_name_list'] = token_name_list
    return one


def create_name_list_by_LexToken(code_obj_list):
    name_list = [''.join(obj.value) if isinstance(obj.value, list) else obj.value for obj in code_obj_list]
    return name_list


def create_token_map_by_action(tokens, action_list):
    """
    :param tokens:
    :param ac_tokens:
    :param action_list: a list of action dict. one action {'type': , 'pos': , 'token': ,},
            position value has consider the gap between tokens
    :return:
    """
    token_to_ac_map = [i for i in range(len(tokens))]

    CHANGE = 0
    INSERT = 1
    DELETE = 2

    bias_list = calculate_action_bias_from_iterative_to_static(action_list)

    for act, bias in zip(action_list, bias_list):
        act_type = act['type']
        act_pos = act['pos']

        if act_type == INSERT:
            real_pos = int(act_pos/2 + bias)
            token_to_ac_map = [token_to_ac_map[i]+1 if i >= real_pos and token_to_ac_map[i] != -1 else token_to_ac_map[i] for i in range(len(token_to_ac_map))]
        elif act_type == DELETE:
            real_pos = int((act_pos-1)/2 + bias)
            token_to_ac_map[real_pos] = -1
            token_to_ac_map = [token_to_ac_map[i]-1 if i > real_pos and token_to_ac_map[i] != -1 else token_to_ac_map[i] for i in range(len(token_to_ac_map))]

    return token_to_ac_map


def create_tokens_error_mask(tokens, action_list):
    token_error_mask = [0 for i in range(len(tokens))]
    bias_list = calculate_action_bias_from_iterative_to_static(action_list)
    try:
        for act, bias in zip(action_list, bias_list):
            act_type = act['type']
            act_pos = act['pos']

            if act_type == INSERT:
                real_pos = int(act_pos/2 + bias)
                if real_pos < len(token_error_mask):
                    token_error_mask[real_pos] = 1
                if (real_pos-1) >= 0:
                    try:
                        token_error_mask[real_pos-1] = 1
                    except Exception as e:
                        print(token_error_mask)
            elif act_type == DELETE:
                real_pos = int((act_pos-1)/2 + bias)
                token_error_mask[real_pos] = 1
            elif act_type == CHANGE:
                real_pos = int((act_pos-1) / 2 + bias)
                token_error_mask[real_pos] = 1
    except Exception as e:
        return None

    return token_error_mask


def create_is_copy_for_ac_tokens(one):
    ac_code_obj = one['ac_code_obj']
    action_token_list = json.loads(one['action_character_list'])

    is_copy_label = [1 for i in range(len(ac_code_obj))]

    for cur_action in action_token_list:
        cur_token_pos = cur_action['token_pos']
        cur_type = cur_action['act_type']

        if cur_type == DELETE:
            is_copy_label[cur_token_pos] = 0
        elif cur_type == CHANGE:
            is_copy_label[cur_token_pos] = 0
    return is_copy_label


def calculate_action_bias_from_iterative_to_static(action_list):

    def cal_token_pos_bias(action_list, cur_action, before_bias_list):
        """
        calculate the bias of iterative action. map iterative action to static tokens.
        In other word, correct the bias causing by previous action
        :param action_list: previous action list iterative
        :param cur_action:
        :param before_bias_list: previous action bias
        :return:
        """
        cur_bias = 0
        cur_token_pos = cur_action['pos']
        for act, bias in zip(action_list, before_bias_list):
            if act['type'] == INSERT and cur_token_pos > (act['pos'] + bias * 2):
                cur_bias -= 1
            elif act['type'] == DELETE and cur_token_pos > (act['pos'] + bias * 2):
                cur_bias += 1
            elif act['type'] == DELETE and cur_action['type'] == DELETE and cur_token_pos >= (act['pos'] + bias * 2):
                cur_bias += 1
        return cur_bias

    bias_list = []
    for i in range(len(action_list)):
        bias_list += [cal_token_pos_bias(action_list[0:i], action_list[i], bias_list)]
    return bias_list



if __name__ == '__main__':
    choosed_list = choose_token_random_batch([100, 20], [[True if i % 10 == 0 else False for i in range(100)], [True if i % 5 == 0 else False for i in range(20)]])
    print(choosed_list)
    # actions = [
    #     # {'type':CHANGE, 'pos': 5},
    #     # {'type':CHANGE, 'pos': 5},
    #     # {'type':INSERT, 'pos': 5},
    #     # {'type':INSERT, 'pos': 5},
    #     {'type':DELETE, 'pos': 4},
    #     {'type':DELETE, 'pos': 4},
    #     # {'type':CHANGE, 'pos': 5},
    # ]
    #
    # token_map = create_token_map_by_action([1 for i in range(10)], actions)
    # print(token_map)
    #
    # error_mask = create_tokens_error_mask([1 for i in range(10)], actions)
    # print(error_mask)