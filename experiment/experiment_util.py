import json

from common.analyse_include_util import extract_include, replace_include_with_blank
from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache
from experiment.parse_xy_util import parse_error_tokens_and_action_map, parse_test_tokens
from read_data.load_data_vocabulary import create_common_error_vocabulary
from read_data.read_experiment_data import read_fake_common_c_error_dataset_with_limit_length

import pandas as pd


MAX_TOKEN_LENGTH = 500


@disk_cache(basename='load_common_error_data', directory=CACHE_DATA_PATH)
def load_common_error_data():
    vocab = create_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'], unk_token='<UNK>', addition_tokens=['<GAP>'])
    train, vaild, test = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)
    train = convert_c_code_fields_to_cpp_fields(train)
    vaild = convert_c_code_fields_to_cpp_fields(vaild)
    test = convert_c_code_fields_to_cpp_fields(test)

    tokenize_fn = tokenize_by_clex_fn()

    parse_param = [vocab, action_list_sorted, tokenize_fn]
    parse_test_param = [vocab, tokenize_fn]

    train_data = parse_error_tokens_and_action_map(train, 'train', *parse_param)
    vaild_data = parse_error_tokens_and_action_map(vaild, 'valid', *parse_param)
    test_data = parse_error_tokens_and_action_map(test, 'test', *parse_param)
    # vaild_data = parse_test_tokens(vaild, 'valid', *parse_test_param)
    # test_data = parse_test_tokens(test, 'test', *parse_test_param)

    train = train.loc[train_data[0].index.values]
    vaild = vaild.loc[vaild_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_code_word_id': train_data[0], 'ac_code_word_id': train_data[1],
                  'token_map': train_data[2], 'error_mask': train_data[3], 'includes': train['includes'], 'is_copy': train_data[4]}
    valid_dict = {'error_code_word_id': vaild_data[0], 'ac_code_word_id': vaild_data[1],
                  'token_map': vaild_data[2], 'error_mask': vaild_data[3], 'includes': vaild['includes'], 'is_copy': vaild_data[4]}
    test_dict = {'error_code_word_id': test_data[0], 'ac_code_word_id': test_data[1], 'token_map': test_data[2],
                 'error_mask': test_data[3], 'includes': test['includes'], 'is_copy': test_data[4]}
    # valid_dict = {'error_code_word_id': vaild_data, 'includes': vaild['includes']}
    # test_dict = {'error_code_word_id': test_data, 'includes': test['includes']}

    # train_data_set = CCodeErrorDataSet(pd.DataFrame(train_dict), vocab, 'train')
    # valid_data_set = CCodeErrorDataSet(pd.DataFrame(valid_dict), vocab, 'all_valid')
    # test_data_set = CCodeErrorDataSet(pd.DataFrame(test_dict), vocab, 'all_test')

    return train_dict, valid_dict, test_dict


# @disk_cache(basename='load_common_error_data_sample_100', directory=CACHE_DATA_PATH)
def load_common_error_data_sample_100():
    vocab = create_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'], unk_token='<UNK>', addition_tokens=['<GAP>'])
    train, vaild, test = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)
    train = convert_c_code_fields_to_cpp_fields(train)
    vaild = convert_c_code_fields_to_cpp_fields(vaild)
    test = convert_c_code_fields_to_cpp_fields(test)

    train = train.sample(100)
    vaild = vaild.sample(100)
    test = test.sample(100)

    tokenize_fn = tokenize_by_clex_fn()

    parse_param = [vocab, action_list_sorted, tokenize_fn]
    parse_test_param = [vocab, tokenize_fn]

    train_data = parse_error_tokens_and_action_map(train, 'train', *parse_param)
    vaild_data = parse_error_tokens_and_action_map(vaild, 'valid', *parse_param)
    test_data = parse_error_tokens_and_action_map(test, 'test', *parse_param)
    # vaild_data = parse_test_tokens(vaild, 'valid', *parse_test_param)
    # test_data = parse_test_tokens(test, 'test', *parse_test_param)

    train = train.loc[train_data[0].index.values]
    vaild = vaild.loc[vaild_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_code_word_id': train_data[0], 'ac_code_word_id': train_data[1],
                  'token_map': train_data[2], 'error_mask': train_data[3], 'includes': train['includes'], 'is_copy': train_data[4]}
    valid_dict = {'error_code_word_id': vaild_data[0], 'ac_code_word_id': vaild_data[1],
                  'token_map': vaild_data[2], 'error_mask': vaild_data[3], 'includes': vaild['includes'], 'is_copy': vaild_data[4]}
    test_dict = {'error_code_word_id': test_data[0], 'ac_code_word_id': test_data[1], 'token_map': test_data[2],
                 'error_mask': test_data[3], 'includes': test['includes'], 'is_copy': test_data[4]}
    # valid_dict = {'error_code_word_id': vaild_data, 'includes': vaild['includes']}
    # test_dict = {'error_code_word_id': test_data, 'includes': test['includes']}

    # train_data_set = CCodeErrorDataSet(pd.DataFrame(train_dict), vocab, 'train')
    # valid_data_set = CCodeErrorDataSet(pd.DataFrame(valid_dict), vocab, 'all_valid')
    # test_data_set = CCodeErrorDataSet(pd.DataFrame(test_dict), vocab, 'all_test')
    # print(train_data[0])

    return train_dict, valid_dict, test_dict


# ---------------------------- convert c code fields to cpp fields ---------------------------------- #
def convert_action_map_to_old_action(actions):
    CHANGE = 0
    INSERT = 1
    DELETE = 2

    def convert_one_action(one_action):
        if one_action['act_type'] == 4:
            one_action['act_type'] = CHANGE
        elif one_action['act_type'] == 3:
            one_action['act_type'] = DELETE
        elif one_action['act_type'] == 1:
            one_action['act_type'] = INSERT
        else:
            print('error_one_action', one_action)
            return None
        return one_action

    new_actions_obj = json.loads(actions)
    old_actions_obj = [convert_one_action(one_action) for one_action in new_actions_obj]
    old_actions_obj = list(filter(lambda x: x is not None, old_actions_obj))
    old_actions = json.dumps(old_actions_obj)
    return old_actions


def convert_c_code_fields_to_cpp_fields(df):
    filter_macro_fn = lambda code: not (code.find('define') != -1 or code.find('defined') != -1 or
                                        code.find('undef') != -1 or code.find('pragma') != -1 or
                                        code.find('ifndef') != -1 or code.find('ifdef') != -1 or
                                        code.find('endif') != -1)
    df['action_character_list'] = df['modify_action_list'].map(convert_action_map_to_old_action)
    df['includes'] = df['similar_code'].map(extract_include)
    df['similar_code_without_include'] = df['similar_code'].map(replace_include_with_blank)
    df['ac_code'] = df['similar_code_without_include']
    df['error_count'] = df['distance']
    df = df[df['similar_code'].map(filter_macro_fn)]
    return df


def action_list_sorted(action_list):
    INSERT = 1
    def sort_key(a):
        bias = 0.5 if a['act_type'] == INSERT else 0
        return a['token_pos'] - bias

    action_list = sorted(action_list, key=sort_key, reverse=True)
    return action_list