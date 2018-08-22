import copy
import itertools
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
from tqdm import tqdm
import torch.multiprocessing as mp

from common.args_util import get_compile_pool
from common.logger import info
from common.torch_util import create_sequence_length_mask, stable_log
from common.util import data_loader, create_effect_keyword_ids_set, PaddedList
from error_generation.find_closest_group_data.token_level_closest_text import \
    calculate_distance_and_action_between_two_code
from experiment.experiment_util import convert_action_map_to_old_action
from model.base_attention import record_is_nan


def generate_action_between_two_code(error_tokens, ac_tokens, max_distance=None, get_value=lambda x: x):
    try:
        distance, action_list = calculate_distance_and_action_between_two_code(error_tokens, ac_tokens,
                                                                      max_distance=max_distance, get_value=get_value)
    except Exception as e:
        distance = -1
        action_list = None
    if max_distance is not None and distance > max_distance:
        distance = -1
        action_list = None
    return distance, action_list


def generate_error_code_from_ac_code_and_action_fn(inner_end_id, begin_id, end_id, vocabulary=None, use_ast=False):
    def generate_error_code_from_ac_code_and_action(input_data, states, action, direct_output=False):
        """

        :param input_data:
        :param states:
        :param action: p1, p2, is_copy, copy_ids, sample_ids. p1 express start position of sample.
        p2 express the end position of sample
        sample position: (p1, p2).
        :return:
        """
        # output_record_list, output_record_probs = create_output_from_actions(states, action, do_sample=True, direct_output=direct_output)
        p1, p2, is_copy, copy_ids, sample_output, sample_output_ids = action

        sample_output_ids_list = sample_output_ids.tolist()
        effect_sample_output_list = []
        for sample in sample_output_ids_list:
            try:
                end_pos = sample.index(inner_end_id)
                sample = sample[:end_pos]
            except ValueError as e:
                pass
            effect_sample_output_list += [sample]
        effect_sample_output_list_length = [len(s) for s in effect_sample_output_list]

        input_seq = input_data['input_seq']
        copy_length = input_data['copy_length']
        input_seq_name = input_data['input_seq_name']

        input_seq = [inp[:cpy] for inp, cpy in zip(input_seq, copy_length)]
        final_output = []
        final_output_name_list = []
        for i, one_input in enumerate(input_seq):
            effect_sample = effect_sample_output_list[i]
            one_output = one_input[1:p1[i] + 1] + effect_sample + one_input[p2[i]:-1]
            final_output += [one_output]
            code_name_list = input_seq_name[i]
            # code name list doesn't have <BEGIN> token .
            # p1 and p2 should sub one to ignore the <BEGIN> token
            effect_sample_name = [vocabulary.id_to_word(word_id) for word_id in effect_sample]
            new_code_name_list = code_name_list[:p1[i]] + effect_sample_name + code_name_list[p2[i] - 1:]
            final_output_name_list += [new_code_name_list]
        # input_data['input_seq_name'] = final_output_name_list

        next_input = [[begin_id] + one + [end_id] for one in final_output]
        # next_input = [next_inp if con else ori_inp for ori_inp, next_inp, con in
        #               zip(input_data['input_seq'], next_input, continue_list)]
        next_input_len = [len(one) for one in next_input]
        final_output = [next_inp[1:-1] for next_inp in next_input]

        input_data['input_seq'] = next_input
        input_data['input_length'] = next_input_len
        input_data['copy_length'] = next_input_len
        input_data['input_seq_name'] = final_output_name_list

        if use_ast:
            res = [create_one_code_res(names, vocabulary) for names in final_output_name_list]
            input_seq, input_length, adjacent_matrix = list(zip(*res))
            input_data['input_seq'] = input_seq
            input_data['adj'] = adjacent_matrix
            input_data['input_length'] = input_length
        return input_data, final_output, effect_sample_output_list_length

    def parse_ast_node(input_seq_name):
        # input_seq_name = [vocabulary.id_to_word(token_id) for token_id in one_final_output]
        # print(' '.join(input_seq_name))
        from c_parser.ast_parser import parse_ast_code_graph
        code_graph = parse_ast_code_graph(input_seq_name)
        input_length = code_graph.graph_length + 2
        in_seq, graph = code_graph.graph
        # begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
        # end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
        input_seq = [begin_id] + [vocabulary.word_to_id(t) for t in in_seq] + [end_id]
        adj = [[a + 1, b + 1] for a, b, _ in graph] + [[b + 1, a + 1] for a, b, _ in graph]
        return input_seq_name, input_seq, adj, input_length

    return generate_error_code_from_ac_code_and_action

def create_output_from_actions_fn(create_or_sample_fn):
    def create_output_from_actions(model_input, model_output, do_sample=False, direct_output=False,
                                   explore_p=0, p2_step_length=3):
        """

        :param model_input:
        :param model_output: p1, p2, is_copy, copy_output, sample_output, compatible_tokens
        sample position: (p1, p1 + p2 + 1).
        :param do_sample:
        :param direct_output:
        :return:
        """
        p1_o, p2_o, is_copy, copy_output, sample_output, output_compatible_tokens = model_output
        if do_sample:
            compatible_tokens = output_compatible_tokens
        else:
            compatible_tokens = model_input[8]

        p1_prob = F.softmax(p1_o, dim=-1)
        p1, p1_log_probs = create_or_sample_fn(p1_prob, explore_p)

        total_mask = create_p2_position_mask(p1_o, p1, p2_step_length)
        p2_o = p2_o.data.masked_fill_(~total_mask, -float('inf'))

        p2_prob = F.softmax(p2_o, dim=-1)
        p2, p2_log_probs = create_or_sample_fn(p2_prob, explore_p)

        if not direct_output:

            is_copy_sigmoid = torch.sigmoid(is_copy)
            not_copy_probs = 1 - is_copy_sigmoid
            is_copy_cat_probs = torch.stack([not_copy_probs, is_copy_sigmoid], dim=-1)
            is_copy, is_copy_log_probs = create_or_sample_fn(is_copy_cat_probs, explore_p)

            # print('copy_output in create_output_from_actions: ', copy_output)
            # record_is_nan(copy_output, 'copy_output in create_output_from_actions')
            copy_output_prob = F.softmax(copy_output, dim=-1)
            sample_output_prob = F.softmax(sample_output, dim=-1)
            # print('copy_output_prob in create_output_from_actions: ', copy_output_prob)
            # record_is_nan(copy_output_prob, 'copy_output_prob in create_output_from_actions')
            copy_output_id, copy_output_log_probs = create_or_sample_fn(copy_output_prob, explore_p)
            # print('copy_output_log_probs in create_output_from_actions: ', copy_output_log_probs)
            # record_is_nan(copy_output_log_probs, 'copy_output_log_probs in create_output_from_actions')
            sample_output_id, sample_output_log_probs = create_or_sample_fn(sample_output_prob, explore_p)

            sample_output = torch.squeeze(torch.gather(compatible_tokens, dim=-1, index=torch.unsqueeze(sample_output_id, dim=-1)), dim=-1)

            input_seq = model_input[1]
            copy_output = torch.gather(input_seq, index=copy_output_id, dim=-1)
        is_copy_byte = is_copy.byte()
        sample_output_ids = torch.where(is_copy_byte, copy_output, sample_output)
        sample_output_ids_log_probs = torch.where(is_copy_byte, copy_output_log_probs, sample_output_log_probs)
        # print(torch.sum(p1_log_probs))
        # print(torch.sum(p2_log_probs))
        # print(torch.sum(is_copy_log_probs))
        # print(torch.sum(copy_output_log_probs))
        # print(torch.sum(sample_output_log_probs))
        # print(torch.sum(sample_output_ids_log_probs))

        return (p1, p2, is_copy, copy_output, sample_output, sample_output_ids), \
               (p1_log_probs, p2_log_probs, is_copy_log_probs, copy_output_log_probs, sample_output_log_probs, sample_output_ids_log_probs)
    return create_output_from_actions


def create_p2_position_mask(p1_o, p1, p2_step_length):
    max_len = p1_o.shape[1]
    batch_size = p1_o.shape[0]
    idxes = torch.arange(0, max_len, out=torch.Tensor(max_len)).to(p1.device).long().unsqueeze(0).expand(batch_size, -1)
    p2_position_max = p1 + p2_step_length + 1
    gt_mask = idxes > torch.unsqueeze(p1, dim=1)
    lt_mask = idxes < torch.unsqueeze(p2_position_max, dim=1)
    total_mask = gt_mask & lt_mask
    res = torch.eq(torch.sum(total_mask, dim=-1), 0)
    total_mask = torch.where(torch.unsqueeze(res, dim=1), torch.ones_like(total_mask).byte(), total_mask)
    return total_mask


def create_or_sample(probs, explore_p):
    if random.random() < explore_p:
        m = Categorical(probs)
        output = m.sample()
        output_log_probs = m.log_prob(output)
    else:
        output_probs, output = torch.topk(probs, dim=-1, k=1)
        output = torch.squeeze(output, dim=-1).data
        output_log_probs = torch.squeeze(stable_log(output_probs), dim=-1)
    return output, output_log_probs


def create_random_sample(probs, explore_p):
    random_probs = torch.ne(probs, 0.0).float()
    m = Categorical(random_probs)
    output = m.sample()
    output_probs = torch.squeeze(torch.gather(probs, index=torch.unsqueeze(output, dim=-1), dim=-1), dim=-1)
    # print('in random sample: ', torch.sum(torch.eq(output_probs, 0.0)))
    output_log_probs = stable_log(output_probs)
    return output, output_log_probs


def create_reward_by_compile(result_list, states, actions, continue_list):
    """
    calculate reward by compile result and states and actions.
    :param result_list:
    :param states:
    :param actions:
    :return:
    """

    new_result_list = []
    for res, con in zip(result_list, continue_list):
        if con and res:
            rew = -1.0
        elif con and not res:
            rew = 1.0
        else:
            rew = 0
        new_result_list += [rew]
    done_list = [False if res else True for res in result_list]
    return new_result_list, done_list


def generate_target_by_code_and_actions(ac_code, actions, prog_id, one_includes, keyword_ids, inner_begin_id,
                                        inner_end_id, use_ast=False, vocabulary=None):
    """
    input argument are all in one records for multi iterate step.
    create all batch data info from ac code and actions .
    :param ac_code: ac code id list with start and end label
    :param actions:
    :param one_includes:
    :param keyword_ids:
    :param inner_begin_id:
    :param inner_end_id:
    :return:
    """
    CHANGE = 0
    INSERT = 1
    DELETE = 2
    from experiment.experiment_util import action_list_sorted_no_reverse
    from experiment.parse_xy_util import action_offset_with_begin_and_end
    from experiment.parse_xy_util import split_actions
    from experiment.parse_xy_util import extract_action_part_start_pos_fn
    from experiment.parse_xy_util import create_sample_error_position_with_iterate
    from common.util import create_effect_keyword_ids_set
    from experiment.parse_xy_util import create_sample_is_copy
    from experiment.parse_xy_util import filter_repeat_action_list
    from experiment.parse_xy_util import create_target_ac_token_id_list
    from experiment.parse_xy_util import create_token_id_input
    from experiment.parse_xy_util import create_token_ids_by_name_fn
    from experiment.parse_xy_util import add_begin_end_label
    from experiment.parse_xy_util import create_one_token_id_by_name_fn

    ac_code_with_label = add_begin_end_label(ac_code)
    ac_code_id_with_label = create_one_token_id_by_name_fn(keyword_voc=vocabulary)(ac_code_with_label)

    action_sort_fn = action_list_sorted_no_reverse
    actions = action_sort_fn(actions)
    actions = filter_repeat_action_list(actions)
    offset_actions = action_offset_with_begin_and_end(actions)
    # offset_actions = actions
    action_part_list = split_actions(offset_actions)
    action_bias_map = {INSERT: 1, DELETE: -1, CHANGE: 0}
    extract_action_start_pos_fn = extract_action_part_start_pos_fn(action_bias_map=action_bias_map)
    ac_pos_list, error_pos_list = extract_action_start_pos_fn(action_part_list)
    one = {'ac_code_name_with_labels': ac_code_with_label,
           'action_part_list': action_part_list,
           'ac_pos_list': ac_pos_list,
           'error_pos_list': error_pos_list}
    one = create_sample_error_position_with_iterate(one)
    token_name_list = one['token_name_list']
    sample_ac_code_list = one['sample_ac_code_list']
    sample_error_code_list = one['sample_error_code_list']

    one = {'token_name_list': token_name_list}
    one = create_token_id_input(one, keyword_voc=vocabulary)
    token_id_list = one['token_id_list']
    token_length_list = one['token_length_list']

    create_token_id_by_name = create_token_ids_by_name_fn(keyword_voc=vocabulary)
    sample_ac_id_list, sample_ac_len_list = create_token_id_by_name(sample_ac_code_list)
    # sample_ac_id_list, sample_ac_len_list = list(zip(*sample_res))

    sample_error_id_list, sample_error_len_list = create_token_id_by_name(sample_error_code_list)
    # sample_error_id_list, sample_error_len_list = list(zip(*sample_res))

    iterate_num = len(token_id_list)

    # sample_ac_len_list = [len(l) for l in sample_ac_id_list]
    # sample_error_len_list = [len(l) for l in sample_error_id_list]

    # keyword_ids = create_effect_keyword_ids_set(keyword_vocab)

    create_input_ids_set_fn = lambda x: list(keyword_ids | set(x[1:-1]))
    sample_mask_list = create_input_ids_set_fn(ac_code_id_with_label)

    one = {'sample_ac_id_list': sample_ac_id_list, 'token_id_list': token_id_list}
    one = create_sample_is_copy(one, keyword_ids)
    is_copy_list = one['is_copy_list']
    copy_pos_list = one['copy_pos_list']

    one = {'token_id_list': token_id_list, 'ac_code_id_with_labels': ac_code_id_with_label}
    one = create_target_ac_token_id_list(one)
    target_ac_token_id_list = one['target_ac_token_id_list']

    # create target
    is_copy_target = [one + [0] for one in is_copy_list]
    copy_target = [one + [-1] for one in copy_pos_list]
    sample_target = [one + [inner_end_id] for one in sample_ac_id_list]
    sample_target = [[t if c == 0 else -1 for c, t in zip(one_is_copy, one_sample)]
                     for one_is_copy, one_sample in zip(is_copy_target, sample_target)]
    p1_target = [error_pos[0] for error_pos in error_pos_list]
    p2_target = [error_pos[1] for error_pos in error_pos_list]

    sample_mask = sorted(sample_mask_list + [inner_end_id])
    sample_mask_dict = {v: i for i, v in enumerate(sample_mask)}
    compatible_tokens = [[sample_mask for _ in range(len(one))]
                         for one in is_copy_target]
    compatible_tokens_length = [[len(tokens) for tokens in one] for one in compatible_tokens]

    sample_small_target = [[sample_mask_dict[t] if c == 0 else -1 for c, t in zip(one_c, one_t)]
                           for one_c, one_t in zip(is_copy_target, sample_target)]
    sample_outputs_length = [len(one) for one in sample_target]

    adjacent_matrix = [0 for _ in range(iterate_num)]
    input_seq = token_id_list
    input_length = [len(ids) for ids in input_seq]
    copy_length = [len(ids) for ids in input_seq]
    target = [[inner_begin_id] + ac_ids + [inner_end_id] for ac_ids in sample_ac_id_list]
    includes = [one_includes for _ in range(iterate_num)]
    id_list = [prog_id for _ in range(iterate_num)]
    token_name_list_output = [names[1:-1] for names in token_name_list]

    if use_ast:
        res = [create_one_code_res(names, vocabulary) for names in token_name_list_output]
        input_seq, input_length, adjacent_matrix = list(zip(*res))

    sample = {'input_seq': input_seq, 'input_length': input_length, 'input_seq_name': token_name_list_output, 'is_copy_target': is_copy_target,
              'copy_target': copy_target, 'copy_length': copy_length, 'sample_target': sample_target,
              'sample_small_target': sample_small_target, 'sample_outputs_length': sample_outputs_length,
              'target': target, 'p1_target': p1_target, 'p2_target': p2_target,
              'compatible_tokens': compatible_tokens, 'compatible_tokens_length': compatible_tokens_length,
              'adj': adjacent_matrix, 'includes': includes, 'id': id_list,
              'full_output_target': target_ac_token_id_list}

    return sample


def create_one_code_res(input_seq_name, vocabulary):
    from c_parser.ast_parser import parse_ast_code_graph
    code_graph = parse_ast_code_graph(input_seq_name)
    input_length = code_graph.graph_length + 2
    in_seq, graph = code_graph.graph
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    input_seq = [begin_id] + [vocabulary.word_to_id(t) for t in in_seq] + [end_id]
    adj = [[a + 1, b + 1] for a, b, _ in graph] + [[b + 1, a + 1] for a, b, _ in graph]
    return input_seq, input_length, adj


def generate_ac_to_error_action_and_create_input_and_target(cur_error_code_names, ac_code_names, max_distance, prog_id,
                                                            one_includes, keyword_ids, inner_begin_id, inner_end_id,
                                                            is_continue, last_sample, use_ast, vocabulary):
    """

    :param cur_error_code_names:
    :param ac_code_names:
    :param max_distance_list:
    :param actions:
    :param prog_id:
    :param one_includes:
    :param keyword_ids:
    :param inner_begin_id:
    :param inner_end_id:
    :return:
    """
    if not is_continue and last_sample is not None:
        return last_sample
    distance, actions = generate_action_between_two_code(cur_error_code_names, ac_code_names,
                                                         max_distance, get_value=lambda x: x)
    if len(actions) == 0:
        from common.action_constants import ActionType
        random_delete = random.randint(0, len(ac_code_names) - 2 - 1)
        actions = [{'act_type': ActionType.DELETE, 'from_char': ac_code_names[random_delete],
                        'to_char': '', 'token_pos': random_delete}]
    actions = convert_action_map_to_old_action(actions)
    sample = generate_target_by_code_and_actions(ac_code_names, actions, prog_id, one_includes, keyword_ids,
                                                 inner_begin_id, inner_end_id, use_ast, vocabulary)
    return sample


def get_one_step_of_sample(multi_iterate_sample_list):
    """
    consist multi_iterate_sample list to multi iterate sample_list
    [{'input': [iterate_one. iterate_two, ...], 'input_two': ...}, {records_two}, ...] ->
    [{'input': [records_one, records_two], 'input_two': ...}, {iterate_two}, ...]
    :param multi_iterate_sample_list:
    :return:
    """
    iterate_time = [len(one_records['input_seq']) for one_records in multi_iterate_sample_list]
    max_iterate_num = max(iterate_time)

    for i in range(max_iterate_num):
        input_data = {}
        for k in multi_iterate_sample_list[0].keys():
            input_data[k] = [one_records[k][min(i, iterate_num-1)]
                             for one_records, iterate_num in zip(multi_iterate_sample_list, iterate_time)]
        yield input_data


def all_output_and_target_evaluate_fn(ignore_token):
    def all_output_and_target_evaluate(model_output, model_target, batch_data):
        p1, p2, is_copy, copy_ids, sample_output, sample_output_ids = model_output
        p1_t, p2_t, is_copy_t, copy_target_t, sample_target_t, sample_small_target_t = model_target

        output_mask = torch.ne(is_copy_t, ignore_token)

        result = torch.eq(p1_t, p1)
        result = result & torch.eq(p2_t, p2)

        # is_copy_ne_count = torch.sum(torch.ne(is_copy, is_copy_t) & output_mask, dim=-1)
        # result = result & torch.eq(is_copy_ne_count, 0)
        #
        # copy_ids_ne_count = torch.sum(torch.ne(copy_ids, copy_target_t) & output_mask, dim=-1)
        # result = result & torch.eq(copy_ids_ne_count, 0)
        #
        # sample_ids_ne_count = torch.sum(torch.ne(sample_output, sample_target_t) & output_mask, dim=-1)
        # result = result & torch.eq(sample_ids_ne_count, 0)

        target_output = torch.LongTensor(PaddedList(batch_data['target'], fill_value=ignore_token)).to(p1_t.device)
        sample_ids_ne_count = torch.sum(torch.ne(sample_output_ids, target_output[:, 1:]) & output_mask, dim=-1)
        result = result & torch.eq(sample_ids_ne_count, 0)
        return result

    return all_output_and_target_evaluate


count = 0
# generate_action_pool = mp.Pool(num_processes)
class GenerateEnvironment(gym.Env):
    def __init__(self, s_model, dataset, batch_size, preprocess_next_input_for_solver_fn,
                 parse_input_batch_data_for_solver_fn, solver_create_next_input_batch_fn, vocabulary,
                 compile_code_ids_fn, extract_includes_fn, create_reward_by_compile_fn, parse_target_batch_data_fn,
                 create_records_all_output_fn, evaluate_output_result_fn,
                 data_radio=1.0, inner_begin_label=None, inner_end_label=None, use_ast=False):
        self.s_model = s_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.preprocess_next_input_for_solver_fn = preprocess_next_input_for_solver_fn
        self.parse_input_batch_data_for_solver_fn = parse_input_batch_data_for_solver_fn
        self.solver_create_next_input_batch_fn = solver_create_next_input_batch_fn
        self.create_records_all_output_fn = create_records_all_output_fn
        self.evaluate_output_result_fn = evaluate_output_result_fn
        self.vocabulary = vocabulary
        self.compile_code_ids_fn = compile_code_ids_fn
        self.extract_includes_fn = extract_includes_fn
        self.data_radio = data_radio
        self.create_reward_by_compile_fn = create_reward_by_compile_fn
        self.parse_target_batch_data_fn = parse_target_batch_data_fn
        self.use_ast = use_ast

        self.continue_list = [True for _ in range(batch_size)]
        self.result_list = [True for _ in range(batch_size)]
        self.ac_batch_data = None
        self.keyword_ids = create_effect_keyword_ids_set(vocabulary)
        self.inner_begin_label = inner_begin_label
        self.inner_end_label = inner_end_label


    def reset(self):
        with tqdm(total=len(self.dataset) * self.data_radio) as pbar:
            for batch_data in data_loader(self.dataset, batch_size=self.batch_size, drop_last=False,
                                          epoch_ratio=self.data_radio):
                self.continue_list = [True for _ in range(self.batch_size)]
                self.result_list = [True for _ in range(self.batch_size)]
                self.last_sample = [None for _ in range(self.batch_size)]
                self.step_action_list = []
                self.ac_batch_data = batch_data.copy()
                yield batch_data
                pbar.update(self.batch_size)

    def step(self, actions, states, states_tensor, file_path, target_file_path):
        with torch.no_grad():
            # calculate p1 and p2 in code without label
            p1 = (actions[0]-1).tolist()
            p2 = (actions[1]-1).tolist()
            ac_action_pos = list(zip(p1, p2))

            # create error code by generate output
            # for i in range(len(self.step_action_list)):
            ori_states = states.copy()
            batch_data, output_ids, effect_sample_output_list_length = \
                self.preprocess_next_input_for_solver_fn(states, states_tensor, actions)

            # recovery not continue records
            for k in batch_data.keys():
                batch_data[k] = [b if c else s for s, b, c, in zip(ori_states[k], batch_data[k], self.continue_list)]
            ori_error_data = batch_data.copy()

            # a = 1
            # for i, c in enumerate(self.continue_list):
            #     if not c:
            #         for k in batch_data.keys():
            #             batch_data[k][i] = ori_states[k][i]

            pool = get_compile_pool()
            batch_size = len(output_ids)

            # generate action between ac code and error code
            max_distance_list = [None for _ in range(batch_size)]
            cur_error_code_ids = batch_data['input_seq']
            cur_error_code_names = batch_data['input_seq_name']
            ac_code_ids = self.ac_batch_data['input_seq']
            ac_code_names = self.ac_batch_data['input_seq_name']
            # generate_args = list(zip(cur_error_code_ids, ac_code_ids, max_distance_list))
            # generate_result_list = list(pool.starmap(generate_action_between_two_code, generate_args))
            # action_list, distance_list = list(zip(*generate_result_list))

            # ac_code_ids = self.ac_batch_data['input_seq']
            prog_id_list = self.ac_batch_data['id']
            includes_list = self.ac_batch_data['includes']
            keyword_ids_list = [self.keyword_ids for _ in range(batch_size)]
            inner_begin_label_list = [self.inner_begin_label for _ in range(batch_size)]
            inner_end_label_list = [self.inner_end_label for _ in range(batch_size)]
            use_ast_list = [self.use_ast for _ in range(batch_size)]
            vocabulary_list = [self.vocabulary for _ in range(batch_size)]
            # generate_target_by_code_and_actions(ac_code_ids, action_list, prog_id_list, includes_list, self.keyword_ids,
            #                                     self.inner_begin_label, self.inner_end_label)

            generate_args = list(zip(cur_error_code_names, ac_code_names, max_distance_list, prog_id_list, includes_list,
                                     keyword_ids_list, inner_begin_label_list, inner_end_label_list, self.continue_list,
                                     self.last_sample, use_ast_list, vocabulary_list))
            # generate_args = [list(args) for args in generate_args]
            generate_result_list = list(pool.starmap(generate_ac_to_error_action_and_create_input_and_target, generate_args))
            # generate_result_list = list(itertools.starmap(generate_ac_to_error_action_and_create_input_and_target, generate_args))
            self.last_sample = generate_result_list
            result = torch.ones(batch_size).byte().to(actions[0].device)
            for one_iterate_sample in get_one_step_of_sample(generate_result_list):
                model_input = self.parse_input_batch_data_for_solver_fn(one_iterate_sample, do_sample=False)
                model_output = self.s_model.forward(*model_input, do_sample=False)
                output_records_list = self.create_records_all_output_fn(model_input, model_output, do_sample=False)
                model_target = self.parse_target_batch_data_fn(one_iterate_sample)
                result_list = self.evaluate_output_result_fn(output_records_list, model_target, one_iterate_sample)
                result = result_list & result

            # generate_action_between_two_code(batch_data, self.ac_batch_data, max_distance=0)
            # for i in range(len(self.step_action_list)):
            #     model_input = self.parse_input_batch_data_for_solver_fn(batch_data, do_sample=True)
            #     model_output = self.s_model.forward(*model_input, do_sample=True)
            #     input_data, final_output, output_records = self.solver_create_next_input_batch_fn(batch_data, model_input, model_output, self.continue_list)
            #
            #     _, self.result_list = self.compile_code_ids_fn(final_output, self.continue_list, self.result_list,
            #                                          vocabulary=self.vocabulary,
            #                                          includes_list=self.extract_includes_fn(input_data),
            #                                          file_path=file_path,
            #                                          target_file_path=target_file_path)

            print_output = False
            global count
            count += 1
            if print_output and count % 10 == 0:
                k = 0
                for ori_code_id, ori_error_id, fin_code_id, res in zip(ori_states['input_seq'], ori_error_data['input_seq'], final_output, self.result_list):
                    if not res:
                        ori_code_id = ori_code_id[1:-1]
                        ori_error_id = ori_error_id[1:-1]

                        ori_code_list = [self.vocabulary.id_to_word(c) for c in ori_code_id]
                        ori_code = ' '.join(ori_code_list)

                        ori_error_list = [self.vocabulary.id_to_word(c) for c in ori_error_id]
                        ori_error_code = ' '.join(ori_error_list)

                        fin_code_list = [self.vocabulary.id_to_word(c) for c in fin_code_id]
                        fin_code = ' '.join(fin_code_list)

                        info('--------------------------- one ------------------------------------')
                        for a in actions:
                            info(str(a[k]))
                        info('ori_code: '+ori_code)
                        info('err_code: '+ori_error_code)
                        info('fin_code: '+fin_code)
                    k += 1

            reward_list, done_list = self.create_reward_by_compile_fn(result, states, actions, self.continue_list)
            self.continue_list = [not done for done in done_list]

            save_list = [reward > 0 for reward in reward_list]
            # done_list = [False for _ in range(len(reward_list))]
        return ori_error_data, reward_list, done_list, {'save_list': save_list,
                                                        'ac_action_pos': ac_action_pos,
                                                        'effect_sample_output_list_length': effect_sample_output_list_length}

    def render(self, mode=''):
        return

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
        return

    def close(self):
        return

    def eval(self):
        self.s_model.eval()

    def train(self):
        self.s_model.train()


def sample_generate_action_fn(create_output_from_actions_fn, calculate_encoder_sample_length_fn,
                              mask_sample_probs_with_length_fn, init_explore_p=0.1, min_explore_p=0.001,
                              decay_step=10000, decay=0.2, p2_step_length=3):
    steps = 0
    explore_p = init_explore_p

    def decay_explore():
        nonlocal explore_p
        if explore_p == min_explore_p:
            return
        explore_p_new = explore_p * decay
        if explore_p_new < min_explore_p:
            explore_p = min_explore_p
        return

    def sample_generate_action(states_tensor, model_output, do_sample=True, direct_output=False):
        nonlocal steps, explore_p
        steps += 1
        if steps % decay_step == 0:
            decay_explore()

        # for i, out in enumerate(model_output):
        #     try:
        #         record_is_nan(out, 'model_output in sample generate action_{}'.format(i))
        #     except Exception as e:
        #         pass

        final_actions, final_actions_probs = create_output_from_actions_fn(states_tensor, model_output,
                                                                           do_sample=do_sample,
                                                                           direct_output=direct_output,
                                                                           explore_p=explore_p,
                                                                           p2_step_length=p2_step_length)
        sample_length = calculate_encoder_sample_length_fn(final_actions)
        final_actions_probs = mask_sample_probs_with_length_fn(final_actions_probs, sample_length)

        return final_actions, (final_actions_probs, )

    return sample_generate_action


def calculate_encoder_sample_length_fn(inner_end_id):
    def calculate_encoder_sample_length(actions):
        """

        :param actions:
        :return:
        """
        p1, p2, is_copy, copy_output, sample_output, sample_output_ids = actions

        sample_output_ids_list = sample_output_ids.tolist()
        sample_length = []
        for sample in sample_output_ids_list:
            try:
                end_pos = sample.index(inner_end_id)
                end_pos += 1
            except ValueError as e:
                end_pos = len(sample)
            sample_length += [end_pos]
        return sample_length
    return calculate_encoder_sample_length


def mask_sample_probs_with_length(action_probs, sample_length):
    """

    :param action_probs: probs with special shape like [batch, seq]
    :param sample_length:
    :return:
    """
    p1_log_probs, p2_log_probs, is_copy_log_probs, copy_output_log_probs, sample_output_log_probs, \
    sample_output_ids_log_probs = action_probs
    # record_is_nan(is_copy_log_probs, 'is_copy_log_probs in mask_sample')
    # record_is_nan(copy_output_log_probs, 'copy_output_log_probs in mask_sample')
    # record_is_nan(sample_output_log_probs, 'sample_output_log_probs in mask_sample')
    if not isinstance(sample_length, torch.Tensor):
        sample_length_tensor = torch.LongTensor(sample_length).to(is_copy_log_probs.device)
    else:
        sample_length_tensor = sample_length
    length_mask_float = create_sequence_length_mask(sample_length_tensor, max_len=is_copy_log_probs.shape[1]).float()

    is_copy_log_probs = is_copy_log_probs * length_mask_float
    sample_output_ids_log_probs = sample_output_ids_log_probs * length_mask_float

    sample_total_log_probs = torch.sum(is_copy_log_probs + sample_output_ids_log_probs, dim=-1) / sample_length_tensor.float()

    # record_is_nan(sample_total_log_probs, 'sample_total_log_probs in mask_sample')
    final_probs = p1_log_probs + p2_log_probs + sample_total_log_probs
    # record_is_nan(final_probs, 'final_probs in mask_sample')
    return final_probs


class GenerateAgent(object):
    def __init__(self, g_model, optimizer, parse_input_batch_data_fn, sample_generate_action_fn, do_sample=True,
                 do_beam_search=False, reward_discount_gamma=0.99, do_normalize=False):
        self.g_model = g_model
        self.optimizer = optimizer
        self.parse_input_batch_data_fn = parse_input_batch_data_fn
        self.do_sample = do_sample
        self.do_beam_search = do_beam_search
        self.sample_generate_action_fn = sample_generate_action_fn
        self.reward_discount_gamma = reward_discount_gamma
        self.do_normalize = do_normalize
        self.g_model.rewards, self.g_model.saved_actions, self.g_model.dones = [], [], []

    def select_action(self, states_tensor):
        # states_tensor = self.parse_input_batch_data_fn(states, do_sample=self.do_sample)
        model_output = self.g_model.forward(*states_tensor, do_sample=self.do_sample, do_beam_search=self.do_beam_search)
        actions, action_probs = self.sample_generate_action_fn(states_tensor, model_output, do_sample=self.do_sample, direct_output=self.do_beam_search)
        self.g_model.saved_actions.append(action_probs)
        return actions

    def add_step_reward(self, reward_list):
        self.g_model.rewards.append(reward_list)

    def get_rewards_sum(self):
        res = np.mean(np.sum(self.g_model.rewards, axis=-1))
        return res

    def discount_rewards(self, model_rewards):
        batch_discounted_rewards = [self.one_discount_rewards(rewards) for rewards in zip(*model_rewards)]
        return batch_discounted_rewards

    def one_discount_rewards(self, one_rewards):
        running_add = 0
        discounted_rewards = []
        for r in one_rewards[::-1]:
            running_add = r + self.reward_discount_gamma * running_add
            discounted_rewards.insert(0, running_add)
        eps = np.finfo(np.float32).eps
        if self.do_normalize:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)
        return discounted_rewards

    def finish_episode(self):
        rewards = self.discount_rewards(self.g_model.rewards)
        rewards = torch.Tensor(rewards).to(self.g_model.saved_actions[0][0].device)
        self.optimizer.zero_grad()
        loss = self.compute_loss(rewards)
        loss.backward()
        self.optimizer.step()
        self.g_model.rewards, self.g_model.saved_actions, self.g_model.dones = [], [], []
        return loss.item()

    def compute_loss(self, rewards):
        # return torch.sum(self.g_model.saved_actions[0][0]) * torch.sum(rewards)
        policy_losses = []
        self.g_model.saved_actions = [list(zip(*s)) for s in self.g_model.saved_actions]
        for one_batch_probs, one_batch_reward in zip(zip(*self.g_model.saved_actions), rewards):
            one_loss = self.one_compute_loss(one_batch_probs, one_batch_reward)
            policy_losses += [one_loss]
        if len(policy_losses) == 1:
            return policy_losses[0]
        return torch.stack(policy_losses).mean()

    def one_compute_loss(self, action_probs, rewards):
        policy_losses = []
        for (probs, ), reward in zip(action_probs, rewards):
            policy_losses.append(- probs * reward)
        if len(policy_losses) == 1:
            return policy_losses[0]
        return torch.stack(policy_losses).sum()

    def train(self):
        self.g_model.train()

    def eval(self):
        self.g_model.eval()


def create_generate_env(s_model, dataset, env_dict):
    env = GenerateEnvironment(s_model, dataset, **env_dict)
    return env


def create_generate_agent(g_model, optimizer, agent_dict):
    agent = GenerateAgent(g_model, optimizer, **agent_dict)
    return agent


class EnvironmentStorage(object):
    def __init__(self):
        pass









