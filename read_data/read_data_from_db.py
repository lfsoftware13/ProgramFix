import sqlite3
import pandas as pd

from common.constants import verdict, langdict, scrapyOJ_DB_PATH, CACHE_DATA_PATH, TRAIN_DATA_DBPATH, \
    ACTUAL_C_ERROR_RECORDS, COMPILE_SUCCESS_DATA_DBPATH, C_COMPILE_SUCCESS_RECORDS, FAKE_C_COMPILE_ERROR_DATA_DBPATH, \
    COMMON_C_ERROR_RECORDS, RANDOM_C_ERROR_RECORDS, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TRAIN, \
    SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TEST, SLK_SAMPLE_COMMON_C_ERROR_RECORDS_VALID, COMMON_DEEPFIX_ERROR_RECORDS
from common.util import disk_cache
from config import DEEPFIX_DB, SLK_SAMPLE_DBPATH, FAKE_DEEPFIX_ERROR_DATA_DBPATH


def merge_and_deal_submit_table(problems_df, submit_df):
    submit_joined_df = submit_df.join(problems_df.set_index('problem_name'), on='problem_name')
    submit_joined_df['time'] = submit_joined_df['time'].str.replace('ms', '').astype('int')
    submit_joined_df['memory'] = submit_joined_df['memory'].str.replace('KB', '').astype('int')
    submit_joined_df['submit_time'] = pd.to_datetime(submit_joined_df['submit_time'])
    submit_joined_df['tags'] = submit_joined_df['tags'].str.split(':')
    submit_joined_df['code'] = submit_joined_df['code'].str.slice(1, -1)
    submit_joined_df['language'] = submit_joined_df['language'].replace(langdict)
    submit_joined_df['status'] = submit_joined_df['status'].replace(verdict)
    return submit_joined_df


def read_all_submit_data(conn: sqlite3.Connection) -> pd.DataFrame:
    problems_df = pd.read_sql('select problem_name, tags from {}'.format('problem'), conn)
    submit_df = pd.read_sql('select * from {}'.format('submit'), conn)
    submit_joined_df = merge_and_deal_submit_table(problems_df, submit_df)
    return submit_joined_df


def read_all_c_data(conn):
    problems_df = pd.read_sql('select problem_name, tags from {}'.format('problem'), conn)
    submit_df = pd.read_sql('select * from {} where language="GNU C"'.format('submit'), conn)
    submit_joined_df = merge_and_deal_submit_table(problems_df, submit_df)
    return submit_joined_df


def read_all_cpp_data(conn):
    problems_df = pd.read_sql('select problem_name, tags from {}'.format('problem'), conn)
    submit_df = pd.read_sql('select * from {} where language="GNU C++"'.format('submit'), conn)
    submit_joined_df = merge_and_deal_submit_table(problems_df, submit_df)
    return submit_joined_df


def read_data(conn, table, condition=None):
    extra_filter = ''
    note = '"'
    if condition is not None:
        extra_filter += ' where '
        condition_str = ['{}{}{}'.format(con[0], con[1], con[2]) for con in condition]
        extra_filter += (' and '.join(condition_str))
    sql = 'select * from {} {}'.format(table, extra_filter)
    data_df = pd.read_sql(sql, conn)
    print('read data sql statment: {}. length:{}'.format(sql, len(data_df.index)))
    return data_df


@disk_cache(basename='read_all_c_records', directory=CACHE_DATA_PATH)
def read_all_c_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(scrapyOJ_DB_PATH), uri=True)
    data_df = read_all_c_data(conn)
    return data_df


@disk_cache(basename='read_all_cpp_records', directory=CACHE_DATA_PATH)
def read_all_cpp_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(scrapyOJ_DB_PATH), uri=True)
    data_df = read_all_cpp_data(conn)
    return data_df


@disk_cache(basename='read_all_submit_records', directory=CACHE_DATA_PATH)
def read_all_submit_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(scrapyOJ_DB_PATH), uri=True)
    data_df = read_all_submit_data(conn)
    return data_df


@disk_cache(basename='read_train_data_all_c_error_records', directory=CACHE_DATA_PATH)
def read_train_data_all_c_error_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(TRAIN_DATA_DBPATH), uri=True)
    data_df = read_data(conn, ACTUAL_C_ERROR_RECORDS)
    return data_df


@disk_cache(basename='read_train_data_effect_all_c_error_records', directory=CACHE_DATA_PATH)
def read_train_data_effect_all_c_error_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(TRAIN_DATA_DBPATH), uri=True)
    condition = [('distance', '>=', 0),
                 ('distance', '<', 10), ]
    data_df = read_data(conn, ACTUAL_C_ERROR_RECORDS, condition)
    return data_df


@disk_cache(basename='read_special_cpp_records', directory=CACHE_DATA_PATH)
def read_special_cpp_records(problem_id, user_id):
    conn = sqlite3.connect("file:{}?mode=ro".format(scrapyOJ_DB_PATH), uri=True)

    problems_df = pd.read_sql('select problem_name, tags from {}'.format('problem'), conn)
    submit_df = pd.read_sql('select * from submit where language="GNU C++" and problem_name="{}" and user_id="{}"'.format(problem_id, user_id), conn)
    submit_joined_df = merge_and_deal_submit_table(problems_df, submit_df)
    print('read special problem id {} user id {} with {} records'.format(problem_id, user_id, len(submit_joined_df)))
    return submit_joined_df


@disk_cache(basename='read_compile_success_c_records', directory=CACHE_DATA_PATH)
def read_compile_success_c_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(COMPILE_SUCCESS_DATA_DBPATH), uri=True)
    data_df = read_data(conn, C_COMPILE_SUCCESS_RECORDS)
    return data_df


@disk_cache(basename='read_fake_common_c_error_records', directory=CACHE_DATA_PATH)
def read_fake_common_c_error_records():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_C_COMPILE_ERROR_DATA_DBPATH), uri=True)
    data_df = read_data(conn, COMMON_C_ERROR_RECORDS)
    return data_df


@disk_cache(basename='read_fake_random_c_error_records', directory=CACHE_DATA_PATH)
def read_fake_random_c_error_records():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_C_COMPILE_ERROR_DATA_DBPATH), uri=True)
    data_df = read_data(conn, RANDOM_C_ERROR_RECORDS)
    return data_df


@disk_cache(basename='read_deepfix_records', directory=CACHE_DATA_PATH)
def read_deepfix_records():
    con = sqlite3.connect("file:{}?mode=ro".format(DEEPFIX_DB), uri=True)
    test_df = pd.read_sql('select * from {}'.format('Code'), con)
    return test_df


@disk_cache(basename='read_slk_grammar_sample_train_records', directory=CACHE_DATA_PATH)
def read_slk_grammar_sample_train_records():
    con = sqlite3.connect("file:{}?mode=ro".format(SLK_SAMPLE_DBPATH), uri=True)
    train_df = pd.read_sql('select * from {}'.format(SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TRAIN), con)
    return train_df


@disk_cache(basename='read_slk_grammar_sample_valid_records', directory=CACHE_DATA_PATH)
def read_slk_grammar_sample_valid_records():
    con = sqlite3.connect("file:{}?mode=ro".format(SLK_SAMPLE_DBPATH), uri=True)
    valid_df = pd.read_sql('select * from {}'.format(SLK_SAMPLE_COMMON_C_ERROR_RECORDS_VALID), con)
    return valid_df


@disk_cache(basename='read_slk_grammar_sample_test_records', directory=CACHE_DATA_PATH)
def read_slk_grammar_sample_test_records():
    con = sqlite3.connect("file:{}?mode=ro".format(SLK_SAMPLE_DBPATH), uri=True)
    test_df = pd.read_sql('select * from {}'.format(SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TEST), con)
    return test_df


@disk_cache(basename='read_fake_common_deepfix_error_records', directory=CACHE_DATA_PATH)
def read_fake_common_deepfix_error_records():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_DEEPFIX_ERROR_DATA_DBPATH), uri=True)
    data_df = read_data(conn, COMMON_DEEPFIX_ERROR_RECORDS)
    return data_df


if __name__ == '__main__':
    df = read_deepfix_records()
    print(len(df))

