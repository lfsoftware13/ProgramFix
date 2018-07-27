from common.pycparser_util import tokenize_by_clex_fn
from read_data.read_filter_data_records import read_distinct_problem_user_compile_success_c_records, \
    read_distinct_problem_user_fake_c_common_records, read_distinct_problem_user_fake_c_random_records, \
    read_distinct_problem_user_c_records, read_deepfix_error_records, read_filter_grammar_sample_test_records, \
    read_filter_grammar_sample_valid_records, read_filter_grammar_sample_train_records, read_deepfix_ac_records
from common.util import disk_cache, filter_length
from common.constants import CACHE_DATA_PATH



def filter_frac(data_df, frac):
    user_count = len(data_df.groupby('user_id').size())
    print('user_count: {}'.format(user_count))
    user_id_list = data_df['user_id'].sample(int(user_count * frac)).tolist()
    print('user_id_list: {}'.format(len(user_id_list)))
    split_df = data_df[data_df.apply(lambda x: x['user_id'] in user_id_list, axis=1, raw=True)]
    print('split_df: {}'.format(len(split_df)))
    # drop_df = data_df[data_df.apply(lambda x: x['user_id'] in user_id_list, axis=1, raw=True)]
    main_df = data_df.drop(split_df.index)
    print('main_df: {}'.format(len(main_df)))
    return main_df, split_df


@disk_cache(basename='read_distinct_problem_user_ac_c99_code_dataset', directory=CACHE_DATA_PATH)
def read_distinct_problem_user_ac_c99_code_dataset():
    data_df = read_distinct_problem_user_compile_success_c_records()

    main_df, test_df = filter_frac(data_df, 0.1)
    train_df, valid_df = filter_frac(main_df, 0.1)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    return train_df, valid_df, test_df


@disk_cache(basename='read_fake_common_c_error_dataset', directory=CACHE_DATA_PATH)
def read_fake_common_c_error_dataset():
    test_df = read_distinct_problem_user_c_records()
    test_df = test_df[test_df['distance'].map(lambda x: 0 < x < 10)]
    data_df = read_distinct_problem_user_fake_c_common_records()
    train_df, valid_df = filter_frac(data_df, 0.1)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    return train_df, valid_df, test_df


@disk_cache(basename='read_fake_random_c_error_dataset', directory=CACHE_DATA_PATH)
def read_fake_random_c_error_dataset():
    test_df = read_distinct_problem_user_c_records()
    test_df = test_df[test_df['distance'].map(lambda x: 0 < x < 10)]
    data_df = read_distinct_problem_user_fake_c_random_records()
    train_df, valid_df = filter_frac(data_df, 0.1)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    return train_df, valid_df, test_df


@disk_cache(basename='read_fake_common_c_error_dataset_with_limit_length', directory=CACHE_DATA_PATH)
def read_fake_common_c_error_dataset_with_limit_length(limit_length=500):
    dfs = read_fake_common_c_error_dataset()
    tokenize_fn = tokenize_by_clex_fn()

    train, valid, test = [filter_length(df, limit_length, tokenize_fn) for df in dfs]
    return train, valid, test


@disk_cache(basename='read_fake_random_c_error_dataset_with_limit_length', directory=CACHE_DATA_PATH)
def read_fake_random_c_error_dataset_with_limit_length(limit_length=500):
    dfs = read_fake_random_c_error_dataset()
    tokenize_fn = tokenize_by_clex_fn()

    train, valid, test = [filter_length(df, limit_length, tokenize_fn) for df in dfs]
    return train, valid, test


@disk_cache(basename='read_deepfix_error_data', directory=CACHE_DATA_PATH)
def read_deepfix_error_data():
    df = read_deepfix_error_records()
    return df


@disk_cache(basename='read_deepfix_ac_data', directory=CACHE_DATA_PATH)
def read_deepfix_ac_data():
    df = read_deepfix_ac_records()
    return df


@disk_cache(basename='read_grammar_sample_error_data', directory=CACHE_DATA_PATH)
def read_grammar_sample_error_data():
    train_df = read_filter_grammar_sample_train_records()
    valid_df = read_filter_grammar_sample_valid_records()
    test_df = read_filter_grammar_sample_test_records()
    return train_df, valid_df, test_df


if __name__ == '__main__':
    # data_df = read_distinct_problem_user_compile_success_c_records()
    # print('all data df', len(data_df))
    # main_df, split_df = filter_frac(data_df, 0.1)
    # print('train_df length: {}, split_df length: {}'.format(len(main_df), len(split_df)))
    # read_distinct_problem_user_ac_c99_code_dataset()
    train_df, valid_df, test_df = read_fake_common_c_error_dataset_with_limit_length(500)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    train_df, valid_df, test_df = read_fake_random_c_error_dataset_with_limit_length(500)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
