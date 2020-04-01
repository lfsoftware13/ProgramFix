import json

import more_itertools

from common.analyse_include_util import extract_include, replace_include_with_blank
from common.constants import CACHE_DATA_PATH, pre_defined_c_tokens
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache, create_effect_keyword_ids_set
from read_data.read_experiment_data import read_fake_common_c_error_dataset_with_limit_length, \
    read_fake_common_deepfix_error_dataset_with_limit_length
from vocabulary.word_vocabulary import load_vocabulary



# common error vocabulary
@disk_cache(basename='read_filter_without_include_ac_token', directory=CACHE_DATA_PATH)
def read_filter_without_include_ac_token():
    train_df, _, _ = read_fake_common_c_error_dataset_with_limit_length(500)
    transform_lextoken_to_token_fn = lambda token_list: [i.value for i in token_list]
    tokenize_fn = tokenize_by_clex_fn()
    parse_tokens = [transform_lextoken_to_token_fn(tokenize_fn(code)) for code in train_df['similar_code']]
    return parse_tokens


@disk_cache(basename='read_modify_action_token', directory=CACHE_DATA_PATH)
def read_modify_action_token():
    train_df, _, _ = read_fake_common_c_error_dataset_with_limit_length(500)
    train_df['modify_action_list'] = train_df['modify_action_list'].map(json.loads)
    extract_to_token_fn = lambda actions: [act['to_char'] for act in actions]
    act_tokens = [extract_to_token_fn(actions) for actions in train_df['modify_action_list']]
    return act_tokens


@disk_cache(basename='get_common_error_vocabulary', directory=CACHE_DATA_PATH)
def get_common_error_vocabulary_set():
    tokens = set(more_itertools.collapse(read_filter_without_include_ac_token()))
    action_tokens = set(more_itertools.collapse(read_modify_action_token()))
    return tokens | action_tokens


@disk_cache(basename='get_common_error_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_common_error_vocabulary_id_map():
    word_list = sorted(get_common_error_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_common_error_vocabulary', directory=CACHE_DATA_PATH)
def create_common_error_vocabulary(begin_tokens, end_tokens, unk_token, addition_tokens=None):
    vocab = load_vocabulary(get_common_error_vocabulary_set, get_common_error_vocabulary_id_map,
                            begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                            addition_tokens=addition_tokens)
    return vocab


# deepfix fake error vocabulary
def get_deepfix_train_ac_tokens_without_includes():
    train_df, _, _ = read_fake_common_deepfix_error_dataset_with_limit_length(500)
    transform_lextoken_to_token_fn = lambda token_list: [i.value for i in token_list]
    tokenize_fn = tokenize_by_clex_fn()
    parse_tokens = [transform_lextoken_to_token_fn(tokenize_fn(code)) for code in train_df['similar_code']]
    return parse_tokens


def get_deepfix_train_error_tokens_without_includes():
    train_df, _, _ = read_fake_common_deepfix_error_dataset_with_limit_length(500)
    transform_lextoken_to_token_fn = lambda token_list: [i.value for i in token_list]
    tokenize_fn = tokenize_by_clex_fn()
    parse_tokens = [transform_lextoken_to_token_fn(tokenize_fn(code)) for code in train_df['code']]
    return parse_tokens


def read_deepfix_modify_action_token():
    train_df, _, _ = read_fake_common_deepfix_error_dataset_with_limit_length(500)
    train_df['modify_action_list'] = train_df['modify_action_list'].map(json.loads)
    extract_to_token_fn = lambda actions: [act['to_char'] for act in actions]
    act_tokens = [extract_to_token_fn(actions) for actions in train_df['modify_action_list']]
    return act_tokens


@disk_cache(basename='get_deepfix_train_token_vocabulary_set', directory=CACHE_DATA_PATH)
def get_deepfix_train_token_vocabulary_set():
    ac_parse_tokens = get_deepfix_train_ac_tokens_without_includes()
    # error_parse_tokens = get_deepfix_train_error_tokens_without_includes()
    action_tokens = read_deepfix_modify_action_token()

    ac_tokens = set(more_itertools.collapse(ac_parse_tokens))
    # err_tokens = set(more_itertools.collapse(error_parse_tokens))
    action_tokens = set(more_itertools.collapse(action_tokens))
    return ac_tokens | action_tokens


@disk_cache(basename='get_deepfix_train_token_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_deepfix_train_token_vocabulary_id_map():
    word_list = sorted(get_deepfix_train_token_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_deepfix_common_error_vocabulary', directory=CACHE_DATA_PATH)
def create_deepfix_common_error_vocabulary(begin_tokens, end_tokens, unk_token, addition_tokens=None):
    vocab = load_vocabulary(get_deepfix_train_token_vocabulary_set, get_deepfix_train_token_vocabulary_id_map,
                            begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                            addition_tokens=addition_tokens)
    return vocab


def load_deepfix_common_error_vocabulary():
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    return vocab


if __name__ == '__main__':
    # res = get_common_error_vocabulary_set()
    # print(type(res), len(res))
    # res = get_common_error_vocabulary_id_map()
    # print(type(res), len(res))
    # vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
    #                                                end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
    #                                                addition_tokens=['<PAD>'])
    import numpy as np
    np.random.seed(100)
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                           end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                           addition_tokens=['<PAD>', '<MASK>'])

    keyword_ids = create_effect_keyword_ids_set(vocab)
    true_keywords = pre_defined_c_tokens
    print(vocab.vocabulary_size, len(true_keywords))

