import json
import multiprocessing
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

from c_parser.buffered_clex import BufferedCLex
from common.analyse_include_util import remove_include
from config import SLK_SAMPLE_DBPATH
from read_data.read_data_from_db import read_all_c_records, read_all_cpp_records, read_slk_grammar_sample_train_records, \
    read_slk_grammar_sample_valid_records, read_slk_grammar_sample_test_records

from common.pycparser_util import init_pycparser, tokenize_by_clex, tokenize_error_count, tokenize_by_clex_fn
from common.util import parallel_map, compile_cpp_code_by_gcc, \
    group_df_to_grouped_list, chunks, compile_c_code_by_gcc, tokenize_cpp_code_by_new_tokenize, init_code, \
    check_ascii_character
from error_generation.find_closest_group_data.token_level_closest_text import find_closest_token_text, init_c_code, \
    save_train_data, \
    filter_repeat_ids, calculate_distance_and_action_between_two_code
from common.constants import verdict, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TRAIN, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_VALID, \
    SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TEST
from common.constants import TRAIN_DATA_DBPATH, ACTUAL_C_ERROR_RECORDS, CPP_TESTCASE_ERROR_RECORDS
from database.database_util import run_sql_select_statment

count = 0


# ---------------------- c compile error ----------------- #
def find_closest_group(one_group: pd.DataFrame):
    sys.setrecursionlimit(5000)
    global count, a_tokenize_error_count, ac_df_length_error, distance_series_error
    current = multiprocessing.current_process()
    if count % 100 == 0:
        print('iteration {} in process {} {}:tokenize_error_count: {}, a_tokenize_error_count: {}, '
              'ac_df_length_error: {}, distance_series_error: {}'.format(count, current.pid, current.name,
                                                                         tokenize_error_count, a_tokenize_error_count,
                                                                         ac_df_length_error, distance_series_error))
    count += 1
    # if not check_group_has_both(one_group):
    #     return None

    c_parser = init_pycparser(lexer=BufferedCLex)
    file_path = '/dev/shm/tmp_file_{}.c'.format(current.pid)
    one_group['gcc_compile_result'] = one_group['code'].apply(compile_c_code_by_gcc, file_path=file_path)
    # one_group['pycparser_result'] = one_group['code'].apply(parse_c_code_by_pycparser, file_path=file_path, c_parser=c_parser, print_exception=False)
    one_group['pycparser_result'] = False
    one_group['code_without_include'] = one_group['code'].map(remove_include).map(lambda x: x.replace('\r', ''))
    one_group['tokenize'] = one_group['code_without_include'].apply(tokenize_by_clex, lexer=c_parser.clex)
    one_group = one_group[one_group['tokenize'].map(lambda x: x is not None)]

    ac_df = one_group[one_group['gcc_compile_result']]
    error_df = one_group[one_group['gcc_compile_result'].map(lambda x: not x)]

    error_df = error_df.apply(find_closest_token_text, axis=1, raw=True, ac_df=ac_df)
    # error_df = error_df[error_df['res'].map(lambda x: x is not None)]
    if 'tokenize' in error_df.columns.values.tolist():
        error_df = error_df.drop(['tokenize'], axis=1)
    if 'tokenize' in ac_df.columns.values.tolist():
        ac_df = ac_df.drop(['tokenize'], axis=1)
    return error_df, ac_df


def check_group_has_both(df):
    hasAC = False
    hasError = False

    res = df['status'].map(lambda x: 1 if x == 1 else 0)
    if np.sum(res) > 0:
        hasAC = True

    res = df['status'].map(lambda x: 1 if x == 7 else 0)
    if np.sum(res) > 0:
        hasError = True
    return hasAC & hasError


def find_c_compile_error_closest_code_main():
    sys.setrecursionlimit(5000)
    data_df = read_all_c_records()
    # data_df = data_df.sample(100000)
    print('finish read code: {}'.format(len(data_df.index)))
    data_df = init_c_code(data_df)
    print('finish init code: {}'.format(len(data_df.index)))
    # data_df = data_df.sample(10000)
    group_list = group_df_to_grouped_list(data_df, 'problem_user_id')
    print('group list length: {}'.format(len(group_list)))
    group_list = list(filter(check_group_has_both, group_list))
    print('after filter both group list length: {}'.format(len(group_list)))
    res = list(parallel_map(8, find_closest_group, group_list))
    # res = [find_closest_group(group) for group in group_list]
    res = list(filter(lambda x: x is not None, res))
    error_df_list, ac_df_list = list(zip(*res))
    total = 0
    for df in error_df_list:
        total += len(df)
    print('final train code: {}'.format(total))
    # save records
    save_train_data(error_df_list, ac_df_list, TRAIN_DATA_DBPATH, ACTUAL_C_ERROR_RECORDS, transform_data_list)


def transform_data_list(one):
    item = []
    item += [one['id']]
    item += [one['submit_url']]
    item += [one['problem_id']]
    item += [one['user_id']]
    item += [one['problem_user_id']]
    item += [one['code']]
    item += [1 if one['gcc_compile_result'] else 0]
    item += [1 if one['pycparser_result'] else 0]
    item += [one['similar_code']] if not one['gcc_compile_result'] else ['']
    action_list = deal_action_type(one['action_list'])
    item += [json.dumps(action_list)] if not one['gcc_compile_result'] else ['']
    item += [int(one['distance'])] if not one['gcc_compile_result'] else [-1]
    return item


# ----------------- cpp semantic testcase error --------------------- #

def filter_reject_records(status):
    error_ver_list = [verdict['WRONG_ANSWER'], verdict['RUNTIME_ERROR'], verdict['TIME_LIMIT_EXCEEDED'],
                      verdict['MEMORY_LIMIT_EXCEEDED']]
    return status in error_ver_list


def filter_ac_records(status):
    return (status == verdict['OK'])


def find_closest_cpp_testcase_group(one_group: pd.DataFrame):
    sys.setrecursionlimit(5000)
    global count, a_tokenize_error_count, ac_df_length_error, distance_series_error
    current = multiprocessing.current_process()
    puid = one_group['problem_user_id'].iloc[0]
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if count % 100 == 0:
        print('[{}] iteration {} in process {} {}:tokenize_error_count: {}, a_tokenize_error_count: {}, '
              'ac_df_length_error: {}, distance_series_error: {}'.format(now_time, count, current.pid, current.name,
                                                                         tokenize_error_count, a_tokenize_error_count,
                                                                         ac_df_length_error, distance_series_error))
        sys.stdout.flush()
        sys.stderr.flush()

    count += 1
    # if not check_group_has_both(one_group):
    #     return None

    file_path = '/dev/shm/tmp_file_{}.cpp'.format(current.pid)
    # file_path = 'tmp_file_{}.cpp'.format(current.pid)
    one_group['gcc_compile_result'] = one_group['code'].apply(compile_cpp_code_by_gcc, file_path=file_path)
    # print(one_group['status'], one_group['gcc_compile_result'])
    # print(one_group.iloc[0]['code'])
    one_group = one_group[one_group['gcc_compile_result'].map(lambda x: x)]
    print('[{}] after compile length: {}. count: {}, process: {}, {}, puid: {}'.format(now_time, len(one_group), count, current.pid, current.name, puid))

    # one_group.loc[:, 'tokenize'] = one_group['code'].map(tokenize_cpp_code_by_new_tokenize)
    one_group = one_group.assign(tokenize=one_group['code'].map(tokenize_cpp_code_by_new_tokenize))
    a_tokenize_error_count += len(one_group['tokenize'].map(lambda x: x is None))
    one_group = one_group[one_group['tokenize'].map(lambda x: x is not None)]
    # one_group = one_group[one_group['tokenize'].map(lambda x: len(x) < 1500)]
    print('[{}] after tokenize length {}. count: {}, process: {}, {}, puid: {}'.format(now_time, len(one_group), count, current.pid, current.name, puid))

    if len(one_group) == 0:
        return None

    ac_df = one_group[one_group['status'].map(filter_ac_records)]
    error_df = one_group[one_group['status'].map(filter_reject_records)]
    print('[{}] after filter ac_df: {}, error_df: {}. count: {}, process: {}, {}, puid: {}'.format(now_time, len(ac_df), len(error_df), count, current.pid, current.name, puid))

    error_df = error_df.apply(find_closest_token_text, axis=1, raw=True, ac_df=ac_df, max_distance=101)
    print('[{}] after find_closest token text error_df: {}. count: {}, process: {}, {}, puid: {}'.format(now_time, len(error_df), count, current.pid, current.name, puid))
    # error_df = error_df[error_df['res'].map(lambda x: x is not None)]
    if 'tokenize' in error_df.columns.values.tolist():
        error_df = error_df.drop(['tokenize'], axis=1)
    if 'tokenize' in ac_df.columns.values.tolist():
        ac_df = ac_df.drop(['tokenize'], axis=1)
    print('[{}] finish iteration {} token text error_df: {}, process: {}, {}, puid: {}'.format(now_time, len(error_df), count, current.pid, current.name, puid))
    return error_df, ac_df


def deal_action_type(action_list):
    for action in action_list:
        action['act_type'] = action['act_type'].value
    return action_list


def transform_cpp_testcase_error_data_list(one):
    item = []
    item += [one['id']]
    item += [one['submit_url']]
    item += [one['problem_id']]
    item += [one['user_id']]
    item += [one['problem_user_id']]
    item += [one['code']]
    item += [1 if one['gcc_compile_result'] else 0]
    item += [one['similar_code']]
    action_list = deal_action_type(one['action_list'])
    item += [json.dumps(action_list)]
    item += [int(one['distance'])]
    return item


def check_both_have_ac_and_testcase_error(df):
    hasAC = False
    hasError = False

    res = df['status'].map(lambda x: 1 if x == 1 else 0)
    if np.sum(res) > 0:
        hasAC = True

    res = df['status'].map(lambda x: 1 if x in [3, 4, 5, 6] else 0)
    if np.sum(res) > 0:
        hasError = True
    return hasAC & hasError


def find_cpp_testcase_error_closest_code_main():
    sys.setrecursionlimit(5000)
    import sqlite3
    try:
        res = run_sql_select_statment(TRAIN_DATA_DBPATH, CPP_TESTCASE_ERROR_RECORDS, 'find_distinct_problem_user_id')
        problem_user_ids = [r[0] for r in res]
    except sqlite3.OperationalError as e:
        problem_user_ids = []
    data_df = read_all_cpp_records()
    # from common.read_data.read_data import read_special_cpp_records
    # data_df = read_special_cpp_records("288E", "10544191")
    print('finish read code: {}'.format(len(data_df.index)))
    data_df = data_df[data_df['status'].map(lambda x: x in [1, 3, 4, 5, 6])]
    # data_df = data_df.sample(100000)
    print('finish filter status: {}'.format(len(data_df.index)))
    data_df = init_c_code(data_df)
    print('finish init code: {}'.format(len(data_df.index)))
    data_df = filter_repeat_ids(data_df, problem_user_ids)
    print('after filter repeat records: {}'.format(len(data_df.index)))
    # data_df = data_df.sample(10000)

    group_list = group_df_to_grouped_list(data_df, 'problem_user_id')
    print('group list length: {}'.format(len(group_list)))
    group_list = list(filter(check_both_have_ac_and_testcase_error, group_list))
    print('after filter both group list length: {}'.format(len(group_list)))
    sys.stdout.flush()

    total = 0
    chunk_count = 0
    for chunk_group in chunks(group_list, 5000):
        total += parallel_and_save_group(chunk_group, chunk_count)
        chunk_count += 1
    print('final train code: {}'.format(total))


def parallel_and_save_group(chunk_group, chunk_count):
    print('chunk {} start groups: {}'.format(chunk_count, len(chunk_group)))
    res = list(parallel_map(10, find_closest_cpp_testcase_group, chunk_group))
    # res = [find_closest_cpp_testcase_group(group) for group in chunk_group]
    res = list(filter(lambda x: x is not None, res))
    if len(res) <= 0:
        return 0
    if len(res) == 1:
        error_df_list = [res[0][0]]
        ac_df_list = [res[0][1]]
    else:
        error_df_list, ac_df_list = list(zip(*res))
    total = 0
    for df in error_df_list:
        total += len(df)
    print('chunk {} end train code: {}'.format(chunk_count, total))
    # save records
    save_train_data(error_df_list, ac_df_list, TRAIN_DATA_DBPATH, CPP_TESTCASE_ERROR_RECORDS,
                    transform_cpp_testcase_error_data_list)
    sys.stdout.flush()
    sys.stderr.flush()
    return total


def calculate_distance_between_two_code_main():

    def filter_code(df):
        df['similar_code'] = df['similar_code'].map(init_code)
        print('data length before check ascii: {}'.format(len(df)))
        df = df[df['similar_code'].map(check_ascii_character)]
        print('data length after check ascii: {}'.format(len(df)))
        df = df[df['code'].map(lambda x: x != '')]
        df['similar_code_without_include'] = df['similar_code'].map(remove_include).map(lambda x: x.replace('\r', ''))
        return df

    def tokenize_code(df):
        tokenize_fn = tokenize_by_clex_fn()
        df['similar_tokenize'] = df['similar_code_without_include'].map(tokenize_fn)
        df = df[df['similar_tokenize'].map(lambda x: x is not None)]
        df['sample_tokenize'] = df['sample_code'].map(tokenize_fn)
        df = df[df['sample_tokenize'].map(lambda x: x is not None)]
        return df

    def cal_distance(one, max_distance=20):
        global count
        count += 1
        if count % 100 == 0:
            print('cal distance: {}'.format(count))
        error_tokenize = one['sample_tokenize']
        ac_tokenize = one['similar_tokenize']
        dis, action_list = calculate_distance_and_action_between_two_code(error_tokenize, ac_tokenize, max_distance=max_distance)
        one['distance'] = dis
        one['modify_action_list'] = action_list
        return one

    def save_data(df:pd.DataFrame, table_name, con:sqlite3.Connection):
        sql = '''update <TABLE> set modify_action_list=?, distance=? where id=?'''
        sql = sql.replace('<TABLE>', table_name)

        save_list = []
        for i, row in df.iterrows():
            row_id = row['id']
            dis = row['distance']
            action_list = row['modify_action_list']
            if dis == -1:
                action_str = ''
            else:
                action_list = deal_action_type(action_list)
                action_str = json.dumps(action_list)
            save_list += [(action_str, dis, row_id)]
            if len(save_list) % 1000 == 0:
                con.executemany(sql, save_list)
                con.commit()
                save_list = []
        con.executemany(sql, save_list)
        con.commit()

    db_path = SLK_SAMPLE_DBPATH
    con = sqlite3.connect(db_path)

    train_df = read_slk_grammar_sample_train_records()
    valid_df = read_slk_grammar_sample_valid_records()
    test_df = read_slk_grammar_sample_test_records()

    # train_df = train_df.sample(100)
    # valid_df = valid_df.sample(100)
    # test_df = test_df.sample(100)

    train_df = filter_code(train_df)
    valid_df = filter_code(valid_df)
    test_df = filter_code(test_df)

    train_df = tokenize_code(train_df)
    valid_df = tokenize_code(valid_df)
    test_df = tokenize_code(test_df)

    train_df = train_df.apply(cal_distance, raw=True, axis=1, max_distance=20)
    valid_df = valid_df.apply(cal_distance, raw=True, axis=1, max_distance=20)
    test_df = test_df.apply(cal_distance, raw=True, axis=1, max_distance=20)

    save_data(train_df, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TRAIN, con)
    save_data(valid_df, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_VALID, con)
    save_data(test_df, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TEST, con)


if __name__ == '__main__':
    calculate_distance_between_two_code_main()
