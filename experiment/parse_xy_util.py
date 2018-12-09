import json
import random

import pandas as pd
import torch

from c_parser.pycparser.pycparser.ply.lex import LexToken
from common.util import create_effect_keyword_ids_set

# ------------------------- produce part token by part position list using tokens tensor --------------------------- #
from experiment.generate_error.generate_error_actions import generate_actions_from_ac_to_error_by_code


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

def parse_test_tokens(df, data_type, keyword_vocab, tokenize_fn, add_begin_end_label=False):
    df['code_tokens'] = df['code'].map(tokenize_fn)
    df = df[df['code_tokens'].map(lambda x: x is not None)].copy()
    df['code_tokens'] = df['code_tokens'].map(list)
    print('after tokenize: ', len(df.index))

    df['code_words'] = df['code_tokens'].map(create_name_list_by_LexToken)
    transform_word_to_id_fn = lambda name_list: keyword_vocab.parse_text([name_list], False)[0]
    df['error_code_word_id'] = df['code_words'].map(transform_word_to_id_fn)
    if add_begin_end_label:
        begin_id = keyword_vocab.word_to_id(keyword_vocab.begin_tokens[0])
        end_id = keyword_vocab.word_to_id(keyword_vocab.end_tokens[0])
        add_fn = lambda x: [begin_id] + x + [end_id]
        add_label_fn = lambda x: [keyword_vocab.begin_tokens[0]] + x + [keyword_vocab.end_tokens[0]]
        df['error_code_word_id'] = df['error_code_word_id'].map(add_fn)
        df['code_words'] = df['code_words'].map(add_label_fn)
    return df['error_code_word_id'], df['code_words']


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
    df['error_code_words'] = df['token_name_list'].map(get_first_fn)

    # do check multi token action
    check_multi_token_action = True
    if check_multi_token_action:
        print('before check_multi_token_action: {}'.format(len(df)))
        check_action_fn = lambda name_list: check_multi_action(name_list, tokenize_fn, has_begin_end_label=False)
        df['multi_action_res'] = df['token_name_list'].map(check_action_fn)
        df = df[df['multi_action_res']]
        print('after check_multi_token_action: {}'.format(len(df)))

    create_token_map_by_action_fn = lambda one: create_token_map_by_action(one['error_code_word_id'], one['action_list'])
    df['token_map'] = df.apply(create_token_map_by_action_fn, raw=True, axis=1)

    create_tokens_error_mask_fn = lambda one: create_tokens_error_mask(one['error_code_word_id'], one['action_list'])
    df['error_mask'] = df.apply(create_tokens_error_mask_fn, raw=True, axis=1)
    df = df[df['error_mask'].map(lambda x: x is not None)].copy()
    print('after error_mask id: ', len(df.index))

    df['is_copy'] = df.apply(create_is_copy_for_ac_tokens, raw=True, axis=1)
    df = df[df['is_copy'].map(lambda x: x is not None)]
    print('after is_copy: ', len(df.index))

    df['pointer_map'] = df.apply(create_one_pointer_map_for_ac_tokens, raw=True, axis=1)

    df['distance'] = df['action_list'].map(len)

    return df['error_code_word_id'], df['ac_code_word_id'], df['token_map'], df['error_mask'], df['is_copy'], \
           df['pointer_map'], df['distance'], df['error_code_words']


def parse_error_tokens_and_action_map_encoder_copy(df, data_type, keyword_vocab, sort_fn=None, tokenize_fn=None,
                                                   inner_begin_id=-1, inner_end_id=-1):
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

    df['is_copy'] = df.apply(create_is_copy_for_ac_tokens_encoder_copy, raw=True, axis=1)

    df['distance'] = df['action_list'].map(len)
    df['ac_code_target_id'] = df.apply(create_encoder_copy_ac_code_target_fn(inner_begin_id, inner_end_id), raw=True,
                                       axis=1)
    df['ac_code_target'] = df['ac_code_target_id'].map(create_encode_copy_model_target_fn(inner_begin_id, inner_end_id))

    return df['error_code_word_id'], df['ac_code_word_id'], df['token_map'], df['error_mask'], df['is_copy'],\
           df['distance'], df['ac_code_target_id'], df['ac_code_target']


CODE_BEGIN = '<BEGIN>'
CODE_END = '<END>'

def parse_iterative_sample_action_error_code(df, data_type, keyword_vocab, sort_fn=None, tokenize_fn=None,
                                             merge_action=True, sequence_output=False):
    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(tokenize_fn)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()
    df['ac_code_obj'] = df['ac_code_obj'].map(list)
    print('after tokenize: ', len(df.index))

    df['ac_code_name'] = df['ac_code_obj'].map(create_name_list_by_LexToken)
    if isinstance(df['action_character_list'].iloc[0], str):
        df['action_token_list'] = df['action_character_list'].map(json.loads)
    else:
        df['action_token_list'] = df['action_character_list']

# sort and filter action list
    if sort_fn is not None:
        df['action_token_list'] = df['action_token_list'].map(sort_fn)

    # filter {{, }}, +++, ;; action
    def filter_special_multi_token_action(action_list):
        special_error_action = ['{{', '}}', '+++', ';;', '[[', ']]', '((', '))', ',,', '===', '&&&', '**', '|||',
                                '%%', '---', '::', '#']
        action_list = [a if a['from_char'] not in special_error_action
                            and a['to_char'] not in special_error_action else None
         for a in action_list]
        action_list = list(filter(lambda x: x != None, action_list))
        return action_list
    df['action_token_list'] = df['action_token_list'].map(filter_special_multi_token_action)
    df = df[df['action_token_list'].map(lambda x: x is not None and len(x) > 0)]

    print('before filter action: {}'.format(len(df['action_token_list'])))
    df['action_token_list'] = df['action_token_list'].map(filter_repeat_action_list)
    df = df[df['action_token_list'].map(lambda x: x is not None)]
    print('after filter action: {}'.format(len(df['action_token_list'])))

    # add begin and end to ac code
    df['ac_code_name_with_labels'] = df['ac_code_name'].map(add_begin_end_label)
    df['ac_code_id_with_labels'] = df['ac_code_name_with_labels'].map(
        create_one_token_id_by_name_fn(keyword_voc=keyword_vocab))
    df['offset_action_token_list'] = df['action_token_list'].map(action_offset_with_begin_and_end)
    print('after action_offset_with_begin_and_end: {}'.format(len(df)))

    df['action_part_list'] = df['offset_action_token_list'].map(create_split_actions_fn(merge_action=merge_action))
    print('after split_actions : {}'.format(len(df)))

    # generate action part pos list
    action_bias_map = {INSERT: 1, DELETE: -1, CHANGE: 0}
    pos_res = df['action_part_list'].map(
        extract_action_part_start_pos_fn(action_bias_map=action_bias_map))
    ac_pos_list, error_pos_list = list(zip(*pos_res))
    df['ac_pos_list'] = list(ac_pos_list)
    # df['ac_pos_list'] = pd.Series(ac_pos_list)
    df['error_pos_list'] = list(error_pos_list)
    # df['error_pos_list'] = pd.Series(error_pos_list)
    print('after extract_action_part_start_pos_fn : {}'.format(len(df)))

    # create input code and sample code according to action part and ac pos and error pos
    df = df.apply(create_sample_error_position_with_iterate, raw=True, axis=1, sequence_output=sequence_output)
    print('after create_sample_error_position_with_iterate : {}'.format(len(df)))

    if sequence_output:
        def set_pos_to_begin_and_end(one):
            ac_pos_list = [(0, len(one['ac_code_name_with_labels']) - 1)]
            error_pos_list = [(0, len(one['token_name_list'][0]) - 1)]
            one['ac_pos_list'] = ac_pos_list
            one['error_pos_list'] = error_pos_list
            return one

        df = df.apply(set_pos_to_begin_and_end, raw=True, axis=1)

    # do check multi token action
    check_multi_token_action = True
    if check_multi_token_action:
        print('before check_multi_token_action: {}'.format(len(df)))
        check_action_fn = lambda name_list: check_multi_action(name_list, tokenize_fn)
        df['multi_action_res'] = df['token_name_list'].map(check_action_fn)
        df = df[df['multi_action_res']]
        print('after check_multi_token_action: {}'.format(len(df)))

    # convert input code to id
    df = df.apply(create_token_id_input, raw=True, axis=1, keyword_voc=keyword_vocab)
    df = df[df['res'].map(lambda x: x is not None)]
    print('after create_token_id_input : {}'.format(len(df)))

    # convert sample code to id
    sample_res = df['sample_ac_code_list'].map(create_token_ids_by_name_fn(keyword_voc=keyword_vocab))
    sample_ac_id_list, sample_ac_len_list = list(zip(*sample_res))
    df['sample_ac_id_list'] = list(sample_ac_id_list)
    df['sample_ac_len_list'] = list(sample_ac_len_list)
    df = df[df['sample_ac_id_list'].map(lambda x: x is not None)]
    print('after sample_ac_id_list : {}'.format(len(df)))

    sample_res = df['sample_error_code_list'].map(create_token_ids_by_name_fn(keyword_voc=keyword_vocab))
    sample_error_id_list, sample_error_len_list = list(zip(*sample_res))
    df['sample_error_id_list'] = list(sample_error_id_list)
    df['sample_error_len_list'] = list(sample_error_len_list)
    df = df[df['sample_error_id_list'].map(lambda x: x is not None)]
    print('after sample_error_id_list : {}'.format(len(df)))

    keyword_ids = create_effect_keyword_ids_set(keyword_vocab)

    create_input_ids_set_fn = lambda x: list(keyword_ids | set(x[1:-1]))
    df['sample_mask_list'] = df['ac_code_id_with_labels'].map(create_input_ids_set_fn)

    df = df.apply(create_sample_is_copy, raw=True, axis=1, keyword_ids=keyword_ids)
    print('after create_sample_is_copy : {}'.format(len(df)))

    df = df.apply(create_target_ac_token_id_list, raw=True, axis=1)

    return df['token_id_list'], df['sample_error_id_list'], df['sample_ac_id_list'], df['ac_pos_list'], \
           df['error_pos_list'], df['ac_code_id_with_labels'], df['is_copy_list'], df['copy_pos_list'], \
           df['sample_mask_list'], df['token_name_list'], df['target_ac_token_id_list'], \
           df['ac_code_name_with_labels']


def check_multi_action(token_names_list, tokenize_fn, has_begin_end_label=True):
    def check_one_multi_action(token_names_without_label):
        code = ' '.join(token_names_without_label)
        new_tokens = tokenize_fn(code)
        if new_tokens is None:
            return False
        new_tokens = [tok.value for tok in new_tokens]
        if len(token_names_without_label) != len(new_tokens):
            return False
        for o, n in zip(token_names_without_label, new_tokens):
            if o != n:
                return False
        return True
    if has_begin_end_label:
        check_res = [check_one_multi_action(names[1:-1]) for names in token_names_list]
    else:
        check_res = [check_one_multi_action(names[:]) for names in token_names_list]
    for r in check_res:
        if not r:
            return False
    return True


def create_target_ac_token_id_list(one):
    target_ac_id_list = one['token_id_list'][1:]
    target_ac_id_list += [one['ac_code_id_with_labels']]
    one['target_ac_token_id_list'] = target_ac_id_list
    return one


def create_sample_is_copy(one, keyword_ids):
    sample_ac_id_list = one['sample_ac_id_list']
    token_id_list = one['token_id_list']

    is_copy_list = []
    copy_pos_list = []
    for sample_ac, token_ids in zip(sample_ac_id_list, token_id_list):
        is_copys = []
        copy_poses = []
        for ac_id in sample_ac:
            if ac_id in keyword_ids:
                copy_pos = -1
                is_copy = 0
            else:
                try:
                    copy_pos = token_ids.index(ac_id)
                    is_copy = 1
                except ValueError as e:
                    copy_pos = -1
                    is_copy = 0
            is_copys = is_copys + [is_copy]
            copy_poses = copy_poses + [copy_pos]
        is_copy_list = is_copy_list + [is_copys]
        copy_pos_list = copy_pos_list + [copy_poses]
    one['is_copy_list'] = is_copy_list
    one['copy_pos_list'] = copy_pos_list
    return one


def create_sample_error_position_with_iterate(one, sequence_output=False):

    def cal_token_pos_bias(action_list, cur_action):
        bias = 0
        cur_token_pos = cur_action['token_pos']
        for act in action_list:
            if act['act_type'] == INSERT and (
                    cur_action['act_type'] == DELETE or cur_action['act_type'] == CHANGE) and cur_token_pos >= act['token_pos']:
                bias += 1
            elif act['act_type'] == INSERT and cur_token_pos > act['token_pos']:
                bias += 1
            elif act['act_type'] == DELETE and cur_token_pos > act['token_pos']:
                bias -= 1
        return bias

    def calculate_last_error_token(last_ac_tokens, action_list):
        token_bias_list = [cal_token_pos_bias(action_list[0:i], action_list[i]) for i in range(len(action_list))]
        for act, token_bias in zip(action_list, token_bias_list):
            ac_type = act['act_type']
            ac_token_pos = act['token_pos']
            real_token_pos = ac_token_pos + token_bias
            if ac_type == INSERT:
                to_char = act['to_char']
                last_ac_tokens = last_ac_tokens[0:real_token_pos] + [to_char] + last_ac_tokens[real_token_pos:]
            elif ac_type == DELETE:
                # from_char = act['from_char']
                last_ac_tokens = last_ac_tokens[0: real_token_pos] + last_ac_tokens[real_token_pos + 1:]
            elif ac_type == CHANGE:
                # from_char = act['from_char']
                to_char = act['to_char']
                last_ac_tokens = last_ac_tokens[0: real_token_pos] + [to_char] + last_ac_tokens[real_token_pos + 1:]
        return last_ac_tokens

    # add begin and end label of tokens and actions
    # ac_code_name_with_labels = one['ac_code_name_with_labels']
    iterate_ac_code_name_with_labels = one['ac_code_name_with_labels']

    action_part_list = one['action_part_list']

    code_token_list = []
    ac_pos_list = one['ac_pos_list']
    error_pos_list = one['error_pos_list']
    sample_ac_code_list = []
    sample_error_code_list = []


    for i in range(len(action_part_list)-1, -1, -1):
        one_action_part = action_part_list[i]
        ac_pos = ac_pos_list[i]
        error_pos = error_pos_list[i]

        iterate_sample_ac_code = iterate_ac_code_name_with_labels[ac_pos[0]+1: ac_pos[1]]
        sample_ac_code_list = [iterate_sample_ac_code] + sample_ac_code_list
        # print('for {}'.format(i))
        # print(one_action_part)
        # print(iterate_ac_code_name_with_labels[ac_pos[0]+1: ac_pos[1]])
        # print(iterate_ac_code_name_with_labels)
        iterate_ac_code_name_with_labels = calculate_last_error_token(iterate_ac_code_name_with_labels, one_action_part)
        # print(iterate_ac_code_name_with_labels)
        # print(iterate_ac_code_name_with_labels[error_pos[0]+1: error_pos[1]])

        iterate_sample_error_code = iterate_ac_code_name_with_labels[error_pos[0]+1: error_pos[1]]
        sample_error_code_list = [iterate_sample_error_code] + sample_error_code_list

        code_token_list = [iterate_ac_code_name_with_labels] + code_token_list
        # ac_pos_list = [ac_pos] + ac_pos_list
        # error_pos_list = [error_pos] + error_pos_list

    one['token_name_list'] = code_token_list
    one['sample_ac_code_list'] = sample_ac_code_list
    one['sample_error_code_list'] = sample_error_code_list

    if sequence_output:
        one['token_name_list'] = [code_token_list[0]]
        one['sample_ac_code_list'] = [one['ac_code_name_with_labels'][1:-1]]
        one['sample_error_code_list'] = [code_token_list[0][1:-1]]

    return one


# tools method
def add_begin_end_label(tokens):
    """
    add begin and end token
    :param tokens:
    :return:
    """
    tokens = [CODE_BEGIN] + tokens + [CODE_END]
    return tokens


def action_offset_with_begin_and_end(action_list):
    for action in action_list:
        action['token_pos'] += 1
    return action_list


def check_neighbor_two_action(before_action, after_action):
    before_type = before_action['act_type']
    before_pos = before_action['token_pos']
    if before_type == INSERT:
        before_pos = before_pos - 0.5
    after_type = after_action['act_type']
    after_pos = after_action['token_pos']
    if after_type != INSERT and 0 < after_pos - before_pos <=1:
        return True
    if after_type == INSERT and 0 < after_pos - before_pos <=1:
        return True
    return False


def create_split_actions_fn(merge_action=True):
    def split_actions(ac_to_error_action_list):
        """

        :param ac_to_error_action_list: action list has been sorted
        :return:
        """
        if not merge_action:
            return [[m] for m in ac_to_error_action_list]
        action_part_list = []
        last_action = None
        for i in range(len(ac_to_error_action_list)-1, -1, -1):
            cur_action = ac_to_error_action_list[i]
            if last_action is None:
                action_part_list = [[cur_action]] + action_part_list
            else:
                is_neighbor = check_neighbor_two_action(cur_action, last_action)
                if is_neighbor:
                    action_part_list[0] = [cur_action] + action_part_list[0]
                else:
                    action_part_list = [[cur_action]] + action_part_list
            last_action = cur_action
        return action_part_list
    return split_actions


def extract_action_part_start_pos_fn(action_bias_map):

    def extract_actions_start_pos(action_parts):
        ac_pos_list = []
        error_pos_list = []
        for i in range(len(action_parts) - 1, -1, -1):
            ac_pos, error_pos = extract_one_part_start_pos(action_parts[i])
            ac_pos_list = [ac_pos] + ac_pos_list
            error_pos_list = [error_pos] + error_pos_list
        return ac_pos_list, error_pos_list

    def extract_one_part_start_pos(one_part):
        start_pos = one_part[0]['token_pos']-1
        if one_part[-1]['act_type'] != INSERT:
            end_ac_pos = one_part[-1]['token_pos'] + 1
        else:
            end_ac_pos = one_part[-1]['token_pos']
        error_bias = [action_bias_map[action['act_type']] for action in one_part]
        end_error_pos = end_ac_pos + sum(error_bias)
        return (start_pos, end_ac_pos), (start_pos, end_error_pos)

    return extract_actions_start_pos


def filter_repeat_action_list(action_token_list):
    token_pos_list = [act['token_pos'] if act['act_type'] != INSERT else -1 for act in action_token_list]
    token_pos_list = list(filter(lambda x: x != -1, token_pos_list))
    has_repeat_action_fn = lambda x: len(set(x)) < len(x)
    if has_repeat_action_fn(token_pos_list):
        return None
    return action_token_list


def parse_output_and_position_map(error_ids, ac_ids, original_distance):
    dis, action_list = generate_actions_from_ac_to_error_by_code(error_ids, ac_ids, max_distance=10)
    if dis >= original_distance or dis <= 0 or dis >= 10:
        return -1, None, None

    is_copy = create_copy_for_ac_tokens(ac_ids, action_list)

    pointer_map = create_pointer_map_for_ac_tokens(ac_ids, action_list, len(error_ids))

    return dis, is_copy, pointer_map


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


def create_token_ids_by_name_fn(keyword_voc):
    def create_token_ids_by_name(token_name_list):
        token_id_list = []
        len_list = []
        for name_list in token_name_list:
            id_list = keyword_voc.parse_text([name_list], False)[0]
            if id_list == None:
                return None, None
            token_id_list.append(id_list)
            len_list.append(len(id_list))
        return token_id_list, len_list
    return create_token_ids_by_name


def create_one_token_id_by_name_fn(keyword_voc):
    def create_one_token_id_by_name(name_list):
        id_list = keyword_voc.parse_text([name_list], False)[0]
        if id_list == None:
            return None
        return id_list
    return create_one_token_id_by_name

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
                    token_error_mask[real_pos-1] = 1
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

    try:
        is_copy_label = create_copy_for_ac_tokens(ac_code_obj, action_token_list)
    except Exception as e:
        return None
    return is_copy_label


def create_copy_for_ac_tokens(ac_code_obj, action_token_list):
    is_copy_label = [1 for i in range(len(ac_code_obj))]

    for cur_action in action_token_list:
        cur_token_pos = cur_action['token_pos']
        cur_type = cur_action['act_type']

        if cur_type == DELETE:
            is_copy_label[cur_token_pos] = 0
        elif cur_type == CHANGE:
            is_copy_label[cur_token_pos] = 0
    return is_copy_label


def create_sample_target_for_encoder_copy_model(ac_code_obj, action_token_list, inner_begin_id, inner_end_id):
    ac_code_length = len(ac_code_obj) + 1
    source_count = [1] * ac_code_length
    source_change = [False] * ac_code_length
    for cur_action in action_token_list:
        cur_token_pos = cur_action['token_pos']
        cur_type = cur_action['act_type']

        if cur_type == DELETE:
            source_count[cur_token_pos] -= 1
            source_count[cur_token_pos] = max(source_count[cur_token_pos], 0)
        elif cur_type == CHANGE:
            pass
        elif cur_type == INSERT:
            source_count[cur_token_pos] += 1
        source_change[cur_token_pos] = True
    res = []
    in_sample = False
    delete_before = False
    for word_id, count, change in zip(ac_code_obj, source_count, source_change):
        if in_sample:
            if change:
                res.append(word_id)
            else:
                in_sample = False
                if delete_before:
                    res.append(word_id)
                    res.append(inner_end_id)
                else:
                    res.append(inner_end_id)
                    res.append(word_id)
        else:
            if change:
                in_sample = True
                res.append(inner_begin_id)
            res.append(word_id)
        if count == 0:
            delete_before = True
        else:
            delete_before = False
    if in_sample:
        res.append(inner_end_id)
    return res


def create_encoder_copy_ac_code_target_fn(inner_begin_id, inner_end_id):
    def f(one):
        ac_code_obj = one['ac_code_word_id']
        action_token_list = json.loads(one['action_character_list'])
        return create_sample_target_for_encoder_copy_model(ac_code_obj, action_token_list, inner_begin_id, inner_end_id)

    return f


def create_encode_copy_model_target_fn(inner_begin_id, inner_end_id):
    def f(x):
        in_sample = False
        res = []
        for t in x:
            if in_sample:
                res.append(t)
                if t == inner_end_id:
                    in_sample = False
            else:
                if t == inner_begin_id:
                    in_sample = True
                    res.append(t)
                else:
                    res.append(-1)
        return res
    return f


def create_is_copy_for_ac_tokens_encoder_copy(one):
    ac_code_obj = one['ac_code_obj']
    action_token_list = json.loads(one['action_character_list'])

    is_copy_label = create_copy_for_encoder(len(one['error_code_word_id'])+1,
                                            len(ac_code_obj)+1, action_token_list)
    return is_copy_label


def create_copy_for_encoder(error_code_length, ac_code_length, action_token_list):
    is_copy_label = [1] * error_code_length
    source_count = [1] * ac_code_length
    source_change = [False] * ac_code_length
    for cur_action in action_token_list:
        cur_token_pos = cur_action['token_pos']
        cur_type = cur_action['act_type']

        if cur_type == DELETE:
            source_count[cur_token_pos] -= 1
            source_count[cur_token_pos] = max(source_count[cur_token_pos], 0)
        elif cur_type == CHANGE:
            pass
        elif cur_type == INSERT:
            source_count[cur_token_pos] += 1
        source_change[cur_token_pos] = True

    c = 0
    for ac_count, ac_change in zip(source_count, source_change):
        if ac_change:
            if ac_count == 0:
                is_copy_label[c] = 0
            else:
                for t in range(c, c+ac_count):
                    is_copy_label[t] = 0
        c += ac_count
    return is_copy_label


def create_one_pointer_map_for_ac_tokens(one):
    ac_code_obj = one['ac_code_obj']
    action_token_list = json.loads(one['action_character_list'])
    error_len = len(one['error_code_word_id'])

    pointer_map = create_pointer_map_for_ac_tokens(ac_code_obj, action_token_list, error_len)
    return pointer_map


def create_pointer_map_for_ac_tokens(ac_code_obj, action_token_list, error_len):
    pointer_map = [i for i in range(len(ac_code_obj))]

    for cur_action in action_token_list:
        cur_token_pos = cur_action['token_pos']
        cur_type = cur_action['act_type']

        if cur_type == DELETE:
            pointer_map[cur_token_pos] = -1
            for i in range(cur_token_pos + 1, len(pointer_map)):
                if pointer_map[i] != -1:
                    pointer_map[i] -= 1
        elif cur_type == INSERT:
            for i in range(cur_token_pos, len(pointer_map)):
                if pointer_map[i] != -1:
                    pointer_map[i] += 1

    last_point = -1
    for i in range(len(pointer_map)):
        if pointer_map[i] == -1:
            pointer_map[i] = last_point + 1
        else:
            last_point = pointer_map[i]
            if last_point + 1 >= error_len:
                last_point = error_len - 2
    return pointer_map


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
    actions = [
        {'act_type':CHANGE, 'token_pos': 5},
        {'act_type':INSERT, 'token_pos': 4},
        {'act_type':DELETE, 'token_pos': 3},
        {'act_type':DELETE, 'token_pos': 7},
        {'act_type':DELETE, 'token_pos': 9},
    ]
    #
    # token_map = create_token_map_by_action([1 for i in range(10)], actions)
    # print(token_map)
    #
    # error_mask = create_tokens_error_mask([1 for i in range(10)], actions)
    # print(error_mask)

    print(create_copy_for_encoder(10, 10, actions))
    target_id = create_sample_target_for_encoder_copy_model(list(range(10)), actions, -2, -3)
    print(target_id)
    print(create_encode_copy_model_target_fn(-2, -3)(target_id))
