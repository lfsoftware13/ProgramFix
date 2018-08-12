import json

from common.analyse_include_util import extract_include, replace_include_with_blank
from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache
from experiment.parse_xy_util import parse_error_tokens_and_action_map, parse_test_tokens, \
    parse_output_and_position_map, parse_error_tokens_and_action_map_encoder_copy, \
    parse_iterative_sample_action_error_code
from read_data.load_data_vocabulary import create_common_error_vocabulary, create_deepfix_common_error_vocabulary
from read_data.read_experiment_data import read_fake_common_c_error_dataset_with_limit_length, read_deepfix_error_data, \
    read_grammar_sample_error_data, read_fake_common_deepfix_error_dataset_with_limit_length

import pandas as pd


MAX_TOKEN_LENGTH = 500



def load_grammar_sample_common_error_data():
    """
    not finish
    :return:
    """
    vocab = create_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'], unk_token='<UNK>',
                                           addition_tokens=['<GAP>'])
    train_df, valid_df, test_df = read_grammar_sample_error_data()
    train_df = convert_c_code_fields_to_cpp_fields(train_df, convert_include=False)
    valid_df = convert_c_code_fields_to_cpp_fields(valid_df, convert_include=False)
    test_df = convert_c_code_fields_to_cpp_fields(test_df, convert_include=False)

    tokenize_fn = tokenize_by_clex_fn()







@disk_cache(basename='load_common_error_data', directory=CACHE_DATA_PATH)
def load_common_error_data(addition_infomation=False):
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
                  'token_map': train_data[2], 'error_mask': train_data[3], 'includes': train['includes'],
                  'is_copy': train_data[4], 'pointer_map': train_data[5], 'distance': train_data[6]}
    valid_dict = {'error_code_word_id': vaild_data[0], 'ac_code_word_id': vaild_data[1],
                  'token_map': vaild_data[2], 'error_mask': vaild_data[3], 'includes': vaild['includes'],
                  'is_copy': vaild_data[4], 'pointer_map': vaild_data[5], 'distance': vaild_data[6]}
    test_dict = {'error_code_word_id': test_data[0], 'ac_code_word_id': test_data[1], 'token_map': test_data[2],
                 'error_mask': test_data[3], 'includes': test['includes'], 'is_copy': test_data[4],
                 'pointer_map': test_data[5], 'distance': test_data[6]}

    if addition_infomation:
        train_dict = add_c_common_code_original_info(data_dict=train_dict, df=train)
        valid_dict = add_c_common_code_original_info(data_dict=valid_dict, df=vaild)
        test_dict = add_c_common_code_original_info(data_dict=test_dict, df=test)

    # valid_dict = {'error_code_word_id': vaild_data, 'includes': vaild['includes']}
    # test_dict = {'error_code_word_id': test_data, 'includes': test['includes']}

    # train_data_set = CCodeErrorDataSet(pd.DataFrame(train_dict), vocab, 'train')
    # valid_data_set = CCodeErrorDataSet(pd.DataFrame(valid_dict), vocab, 'all_valid')
    # test_data_set = CCodeErrorDataSet(pd.DataFrame(test_dict), vocab, 'all_test')

    return train_dict, valid_dict, test_dict


# @disk_cache(basename='load_common_error_data_sample_100', directory=CACHE_DATA_PATH)
def load_common_error_data_sample_100(addition_infomation=False):
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
                  'token_map': train_data[2], 'error_mask': train_data[3], 'includes': train['includes'],
                  'is_copy': train_data[4], 'pointer_map': train_data[5], 'distance': train_data[6]}
    valid_dict = {'error_code_word_id': vaild_data[0], 'ac_code_word_id': vaild_data[1],
                  'token_map': vaild_data[2], 'error_mask': vaild_data[3], 'includes': vaild['includes'],
                  'is_copy': vaild_data[4], 'pointer_map': vaild_data[5], 'distance': vaild_data[6]}
    test_dict = {'error_code_word_id': test_data[0], 'ac_code_word_id': test_data[1], 'token_map': test_data[2],
                 'error_mask': test_data[3], 'includes': test['includes'], 'is_copy': test_data[4],
                 'pointer_map': test_data[5], 'distance': test_data[6]}

    if addition_infomation:
        train_dict = add_c_common_code_original_info(data_dict=train_dict, df=train)
        valid_dict = add_c_common_code_original_info(data_dict=valid_dict, df=vaild)
        test_dict = add_c_common_code_original_info(data_dict=test_dict, df=test)
    # valid_dict = {'error_code_word_id': vaild_data, 'includes': vaild['includes']}
    # test_dict = {'error_code_word_id': test_data, 'includes': test['includes']}

    # train_data_set = CCodeErrorDataSet(pd.DataFrame(train_dict), vocab, 'train')
    # valid_data_set = CCodeErrorDataSet(pd.DataFrame(valid_dict), vocab, 'all_valid')
    # test_data_set = CCodeErrorDataSet(pd.DataFrame(test_dict), vocab, 'all_test')
    # print(train_data[0])

    return train_dict, valid_dict, test_dict


@disk_cache(basename='load_common_error_data_with_encoder_copy', directory=CACHE_DATA_PATH)
def load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    vocab = create_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'], unk_token='<UNK>', addition_tokens=['<GAP>'])
    train, vaild, test = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)
    train = convert_c_code_fields_to_cpp_fields(train)
    vaild = convert_c_code_fields_to_cpp_fields(vaild)
    test = convert_c_code_fields_to_cpp_fields(test)

    tokenize_fn = tokenize_by_clex_fn()

    parse_param = [vocab, action_list_sorted, tokenize_fn, inner_begin_id, inner_end_id]
    parse_test_param = [vocab, tokenize_fn]

    train_data = parse_error_tokens_and_action_map_encoder_copy(train, 'train', *parse_param)
    vaild_data = parse_error_tokens_and_action_map_encoder_copy(vaild, 'valid', *parse_param)
    test_data = parse_error_tokens_and_action_map_encoder_copy(test, 'test', *parse_param)
    # vaild_data = parse_test_tokens(vaild, 'valid', *parse_test_param)
    # test_data = parse_test_tokens(test, 'test', *parse_test_param)

    train = train.loc[train_data[0].index.values]
    vaild = vaild.loc[vaild_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_code_word_id': train_data[0], 'ac_code_word_id': train_data[1],
                  'token_map': train_data[2], 'error_mask': train_data[3], 'includes': train['includes'],
                  'is_copy': train_data[4], 'distance': train_data[5], 'ac_code_target_id': train_data[6],
                  'ac_code_target': train_data[7]}
    valid_dict = {'error_code_word_id': vaild_data[0], 'ac_code_word_id': vaild_data[1],
                  'token_map': vaild_data[2], 'error_mask': vaild_data[3], 'includes': vaild['includes'],
                  'is_copy': vaild_data[4], 'distance': vaild_data[5], 'ac_code_target_id': vaild_data[6],
                  'ac_code_target': vaild_data[7]}
    test_dict = {'error_code_word_id': test_data[0], 'ac_code_word_id': test_data[1], 'token_map': test_data[2],
                 'error_mask': test_data[3], 'includes': test['includes'], 'is_copy': test_data[4],
                 'distance': test_data[5], 'ac_code_target_id': test_data[6],
                 'ac_code_target': test_data[7]}
    # valid_dict = {'error_code_word_id': vaild_data, 'includes': vaild['includes']}
    # test_dict = {'error_code_word_id': test_data, 'includes': test['includes']}

    # train_data_set = CCodeErrorDataSet(pd.DataFrame(train_dict), vocab, 'train')
    # valid_data_set = CCodeErrorDataSet(pd.DataFrame(valid_dict), vocab, 'all_valid')
    # test_data_set = CCodeErrorDataSet(pd.DataFrame(test_dict), vocab, 'all_test')
    # print(train_data[0])

    return train_dict, valid_dict, test_dict


def load_common_error_data_sample_with_encoder_copy_100(inner_begin_id, inner_end_id):
    vocab = create_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'], unk_token='<UNK>', addition_tokens=['<GAP>'])
    train, vaild, test = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)
    train = convert_c_code_fields_to_cpp_fields(train)
    vaild = convert_c_code_fields_to_cpp_fields(vaild)
    test = convert_c_code_fields_to_cpp_fields(test)

    train = train.sample(100)
    vaild = vaild.sample(100)
    test = test.sample(100)

    tokenize_fn = tokenize_by_clex_fn()

    parse_param = [vocab, action_list_sorted, tokenize_fn, inner_begin_id, inner_end_id]
    parse_test_param = [vocab, tokenize_fn]

    train_data = parse_error_tokens_and_action_map_encoder_copy(train, 'train', *parse_param)
    vaild_data = parse_error_tokens_and_action_map_encoder_copy(vaild, 'valid', *parse_param)
    test_data = parse_error_tokens_and_action_map_encoder_copy(test, 'test', *parse_param)
    # vaild_data = parse_test_tokens(vaild, 'valid', *parse_test_param)
    # test_data = parse_test_tokens(test, 'test', *parse_test_param)

    train = train.loc[train_data[0].index.values]
    vaild = vaild.loc[vaild_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_code_word_id': train_data[0], 'ac_code_word_id': train_data[1],
                  'token_map': train_data[2], 'error_mask': train_data[3], 'includes': train['includes'],
                  'is_copy': train_data[4], 'distance': train_data[5], 'ac_code_target_id': train_data[6],
                  'ac_code_target': train_data[7]}
    valid_dict = {'error_code_word_id': vaild_data[0], 'ac_code_word_id': vaild_data[1],
                  'token_map': vaild_data[2], 'error_mask': vaild_data[3], 'includes': vaild['includes'],
                  'is_copy': vaild_data[4], 'distance': vaild_data[5], 'ac_code_target_id': vaild_data[6],
                  'ac_code_target': vaild_data[7]}
    test_dict = {'error_code_word_id': test_data[0], 'ac_code_word_id': test_data[1], 'token_map': test_data[2],
                 'error_mask': test_data[3], 'includes': test['includes'], 'is_copy': test_data[4],
                 'distance': test_data[5], 'ac_code_target_id': test_data[6],
                 'ac_code_target': test_data[7]}
    # valid_dict = {'error_code_word_id': vaild_data, 'includes': vaild['includes']}
    # test_dict = {'error_code_word_id': test_data, 'includes': test['includes']}

    # train_data_set = CCodeErrorDataSet(pd.DataFrame(train_dict), vocab, 'train')
    # valid_data_set = CCodeErrorDataSet(pd.DataFrame(valid_dict), vocab, 'all_valid')
    # test_data_set = CCodeErrorDataSet(pd.DataFrame(test_dict), vocab, 'all_test')
    # print(train_data[0])

    return train_dict, valid_dict, test_dict


def add_c_common_code_original_info(data_dict, df):
    data_dict['id'] = df['id']
    data_dict['problem_id'] = df['problem_id']
    data_dict['user_id'] = df['user_id']
    data_dict['problem_user_id'] = df['problem_user_id']
    data_dict['code'] = df['code']
    data_dict['similar_code'] = df['similar_code']
    data_dict['original_modify_action_list'] = df['modify_action_list']
    data_dict['original_distance'] = df['distance']
    return data_dict


@disk_cache(basename='load_deepfix_error_data', directory=CACHE_DATA_PATH)
def load_deepfix_error_data():
    vocab = create_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'], unk_token='<UNK>',
                                           addition_tokens=['<GAP>'])
    df = read_deepfix_error_data()
    df = convert_deepfix_to_c_code(df)

    tokenize_fn = tokenize_by_clex_fn()
    parse_test_param = [vocab, tokenize_fn]
    df_data = parse_test_tokens(df, 'deepfix', *parse_test_param)

    df = df.loc[df_data.index.values]

    deepfix_dict = {'error_code_word_id': df_data, 'includes': df['includes'], 'distance': df['errorcount']}
    return deepfix_dict


# @disk_cache(basename='load_fake_deepfix_dataset_iterate_error_data_sample_100', directory=CACHE_DATA_PATH)
def load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=False):
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])

    train, valid, test = read_fake_common_deepfix_error_dataset_with_limit_length(500)

    train = train.sample(100)
    valid = valid.sample(100)
    test = test.sample(100)

    train = convert_c_code_fields_to_cpp_fields(train, convert_include=False)
    valid = convert_c_code_fields_to_cpp_fields(valid, convert_include=False)
    test = convert_c_code_fields_to_cpp_fields(test, convert_include=False)

    tokenize_fn = tokenize_by_clex_fn()
    parse_fn = parse_iterative_sample_action_error_code
    parse_param = [vocab, action_list_sorted_no_reverse, tokenize_fn]

    train_data = parse_fn(train, 'train', *parse_param)
    valid_data = parse_fn(valid, 'valid', *parse_param)
    test_data = parse_fn(test, 'test', *parse_param)

    train = train.loc[train_data[0].index.values]
    valid = valid.loc[valid_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_token_id_list': train_data[0], 'sample_error_id_list': train_data[1],
                  'sample_ac_id_list': train_data[2], 'ac_pos_list': train_data[3],
                  'error_pos_list': train_data[4], 'includes': train['includes'],
                  'distance': train['distance'], 'ac_code_ids': train_data[5],
                  'is_copy_list': train_data[6], 'copy_pos_list': train_data[7], 'sample_mask_list': train_data[8],
                  'error_token_name_list': train_data[9]}
    valid_dict = {'error_token_id_list': valid_data[0], 'sample_error_id_list': valid_data[1],
                  'sample_ac_id_list': valid_data[2], 'ac_pos_list': valid_data[3],
                  'error_pos_list': valid_data[4], 'includes': valid['includes'],
                  'distance': valid['distance'], 'ac_code_ids': valid_data[5],
                  'is_copy_list': valid_data[6], 'copy_pos_list': valid_data[7], 'sample_mask_list': valid_data[8],
                  'error_token_name_list': valid_data[9]}
    test_dict = {'error_token_id_list': test_data[0], 'sample_error_id_list': test_data[1],
                 'sample_ac_id_list': test_data[2], 'ac_pos_list': test_data[3],
                 'error_pos_list': test_data[4], 'includes': test['includes'],
                 'distance': test['distance'], 'ac_code_ids': test_data[5],
                 'is_copy_list': test_data[6], 'copy_pos_list': test_data[7], 'sample_mask_list': test_data[8],
                 'error_token_name_list': test_data[9]}

    if do_flatten:
        train_dict = flatten_iterative_data(train_dict)
        valid_dict = flatten_iterative_data(valid_dict)
        test_dict = flatten_iterative_data(test_dict)

    return train_dict, valid_dict, test_dict


# @disk_cache(basename='load_fake_deepfix_dataset_iterate_error_data', directory=CACHE_DATA_PATH)
def load_fake_deepfix_dataset_iterate_error_data(do_flatten=False):
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])

    train, valid, test = read_fake_common_deepfix_error_dataset_with_limit_length(500)

    train = convert_c_code_fields_to_cpp_fields(train, convert_include=False)
    valid = convert_c_code_fields_to_cpp_fields(valid, convert_include=False)
    test = convert_c_code_fields_to_cpp_fields(test, convert_include=False)

    tokenize_fn = tokenize_by_clex_fn()
    parse_fn = parse_iterative_sample_action_error_code
    parse_param = [vocab, action_list_sorted_no_reverse, tokenize_fn]

    train_data = parse_fn(train, 'train', *parse_param)
    valid_data = parse_fn(valid, 'valid', *parse_param)
    test_data = parse_fn(test, 'test', *parse_param)

    train = train.loc[train_data[0].index.values]
    valid = valid.loc[valid_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_token_id_list': train_data[0], 'sample_error_id_list': train_data[1],
                  'sample_ac_id_list': train_data[2], 'ac_pos_list': train_data[3],
                  'error_pos_list': train_data[4], 'includes': train['includes'],
                  'distance': train['distance'], 'ac_code_ids': train_data[5],
                  'is_copy_list': train_data[6], 'copy_pos_list': train_data[7], 'sample_mask_list': train_data[8],
                  'error_token_name_list': train_data[9]}
    valid_dict = {'error_token_id_list': valid_data[0], 'sample_error_id_list': valid_data[1],
                  'sample_ac_id_list': valid_data[2], 'ac_pos_list': valid_data[3],
                  'error_pos_list': valid_data[4], 'includes': valid['includes'],
                  'distance': valid['distance'], 'ac_code_ids': valid_data[5],
                  'is_copy_list': valid_data[6], 'copy_pos_list': valid_data[7], 'sample_mask_list': valid_data[8],
                  'error_token_name_list': valid_data[9]}
    test_dict = {'error_token_id_list': test_data[0], 'sample_error_id_list': test_data[1],
                  'sample_ac_id_list': test_data[2], 'ac_pos_list': test_data[3],
                  'error_pos_list': test_data[4], 'includes': test['includes'],
                  'distance': test['distance'], 'ac_code_ids': test_data[5],
                 'is_copy_list': test_data[6], 'copy_pos_list': test_data[7], 'sample_mask_list': test_data[8],
                 'error_token_name_list': test_data[9]}

    if do_flatten:
        train_dict = flatten_iterative_data(train_dict)
        valid_dict = flatten_iterative_data(valid_dict)
        test_dict = flatten_iterative_data(test_dict)

    return train_dict, valid_dict, test_dict


# @disk_cache(basename='load_deepfix_error_data_for_iterate', directory=CACHE_DATA_PATH)
def load_deepfix_error_data_for_iterate():
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    df = read_deepfix_error_data()
    df = convert_deepfix_to_c_code(df)

    tokenize_fn = tokenize_by_clex_fn()
    parse_test_param = [vocab, tokenize_fn, True]
    df_data = parse_test_tokens(df, 'deepfix', *parse_test_param)

    df = df.loc[df_data.index.values]

    deepfix_dict = {'error_token_id_list': df_data, 'includes': df['includes'], 'distance': df['errorcount'],
                    'error_token_name_list': df_data}
    return deepfix_dict


def flatten_iterative_data(data_dict):
    flatten_dict = {}
    records = []
    for i in range(len(data_dict['error_token_id_list'])):
        error_token_id_list = data_dict['error_token_id_list'].iloc[i]
        sample_error_id_list = data_dict['sample_error_id_list'].iloc[i]
        sample_ac_id_list = data_dict['sample_ac_id_list'].iloc[i]
        ac_pos_list = data_dict['ac_pos_list'].iloc[i]
        error_pos_list = data_dict['error_pos_list'].iloc[i]
        includes = data_dict['includes'].iloc[i]
        distance = data_dict['distance'].iloc[i]
        ac_code_ids = data_dict['ac_code_ids'].iloc[i]
        is_copy_list = data_dict['is_copy_list'].iloc[i]
        copy_pos_list = data_dict['copy_pos_list'].iloc[i]
        sample_mask_list = data_dict['sample_mask_list'].iloc[i]
        error_token_name_list = data_dict['error_token_name_list'].iloc[i]
        c = 0
        for error_token_id, sample_error_id, sample_ac_id, ac_pos, error_pos, is_copy, copy_pos, error_token_name in \
            zip(error_token_id_list, sample_error_id_list, sample_ac_id_list, ac_pos_list, error_pos_list,
                is_copy_list, copy_pos_list, error_token_name_list):
            if (c+1) < len(error_token_id_list):
                target_ac_token_id = error_token_id_list[c+1]
            else:
                target_ac_token_id = ac_code_ids
            one = (error_token_id, sample_error_id, sample_ac_id, ac_pos, error_pos, includes, distance, ac_code_ids,
                   is_copy, copy_pos, sample_mask_list, target_ac_token_id, error_token_name)
            records += [one]
            c += 1

    for key, v in zip(['error_token_id_list', 'sample_error_id_list', 'sample_ac_id_list', 'ac_pos_list',
                       'error_pos_list', 'includes', 'distance', 'ac_code_ids', 'is_copy_list',
                       'copy_pos_list', 'sample_mask_list', 'target_ac_token_id_list',
                       'error_token_name_list'], zip(*records)):
        flatten_dict[key] = pd.Series(v)
    return flatten_dict


# ---------------------------------- addition train dataset --------------------------------------- #
def create_addition_error_data(records):
    error_code_word_ids = [rec['error_code_word_id'] for rec in records]
    ac_code_word_ids = [rec['ac_code_word_id'] for rec in records]
    includes = [rec['includes'] for rec in records]
    res = [parse_output_and_position_map(rec['error_code_word_id'], rec['ac_code_word_id'], rec['original_distance'])
           for rec in records]
    distances, is_copys, pointer_maps = list(zip(*res))
    addition_dict = {'error_code_word_id': error_code_word_ids, 'ac_code_word_id': ac_code_word_ids,
                     'includes': includes, 'is_copy': is_copys, 'pointer_map': pointer_maps, 'distance': distances}
    return addition_dict


def create_copy_addition_data(ac_code_ids, includes):
    ac_code_word_ids = ac_code_ids
    distances = [0 for i in range(len(ac_code_ids))]
    is_copys = [[1 for i in range(len(ac_ids))] for ac_ids in ac_code_ids]
    pointer_maps = [[i for i in range(len(ac_ids))] for ac_ids in ac_code_ids]
    addition_dict = {'error_code_word_id': ac_code_word_ids, 'ac_code_word_id': ac_code_word_ids,
                     'includes': includes, 'is_copy': is_copys, 'pointer_map': pointer_maps, 'distance': distances}
    return addition_dict


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


def convert_c_code_fields_to_cpp_fields(df, convert_include=True):
    filter_macro_fn = lambda code: not (code.find('define') != -1 or code.find('defined') != -1 or
                                        code.find('undef') != -1 or code.find('pragma') != -1 or
                                        code.find('ifndef') != -1 or code.find('ifdef') != -1 or
                                        code.find('endif') != -1)
    df['action_character_list'] = df['modify_action_list'].map(convert_action_map_to_old_action)
    if convert_include:
        df['includes'] = df['similar_code'].map(extract_include)
    df['similar_code_without_include'] = df['similar_code'].map(replace_include_with_blank)
    df['ac_code'] = df['similar_code_without_include']
    df['error_count'] = df['distance']
    df = df[df['similar_code'].map(filter_macro_fn)]
    return df


def convert_deepfix_to_c_code(df):
    filter_macro_fn = lambda code: not (code.find('define') != -1 or code.find('defined') != -1 or
                                        code.find('undef') != -1 or code.find('pragma') != -1 or
                                        code.find('ifndef') != -1 or code.find('ifdef') != -1 or
                                        code.find('endif') != -1)
    df['includes'] = df['code'].map(extract_include)
    df['code_with_include'] = df['code']
    df['code'] = df['code_with_include'].map(replace_include_with_blank)
    df = df[df['code'].map(filter_macro_fn)]
    return df


def action_list_sorted(action_list, reverse=True):
    INSERT = 1
    def sort_key(a):
        bias = 0.5 if a['act_type'] == INSERT else 0
        return a['token_pos'] - bias

    action_list = sorted(action_list, key=sort_key, reverse=reverse)
    return action_list

def action_list_sorted_no_reverse(action_list):
    return action_list_sorted(action_list, reverse=False)


if __name__ == '__main__':
    # data_dict = load_deepfix_error_data()
    # print(data_dict['error_code_word_id'].iloc[0])
    # print(data_dict['includes'].iloc[0])
    # print(data_dict['distance'].iloc[0])
    # print(len(data_dict['error_code_word_id']))

    train_dict, valid_dict, test_dict = load_fake_deepfix_dataset_iterate_error_data()
    print(len(train_dict['error_token_id_list']))
    print(len(valid_dict['error_token_id_list']))
    print(len(test_dict['error_token_id_list']))


