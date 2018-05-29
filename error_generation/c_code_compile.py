from read_data.read_data_from_db import read_all_c_records
from common.util import compile_c_code_by_gcc, chunks, parallel_map, reverse_dict
from error_generation.find_closest_group_data.token_level_closest_text import init_c_code
from common.constants import COMPILE_SUCCESS_DATA_DBPATH, C_COMPILE_SUCCESS_RECORDS, verdict, langdict
from database.database_util import create_table, insert_items

import multiprocessing
import time
import numpy as np
import sys


def do_c_compile(one, file_path):
    code = one['code']
    if code is None or code == '':
        return False

    res = compile_c_code_by_gcc(code, file_path)
    return res


count = 0
def compile_map_fn(data_df):
    global count
    count += 1
    current = multiprocessing.current_process()
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if count % 1 == 0:
        print('[{}] iteration {} in process {} {}:'.format(now_time, count, current.pid, current.name))
        sys.stdout.flush()
        sys.stderr.flush()

    current = multiprocessing.current_process()
    file_path = '/dev/shm/tmp_file_{}.c'.format(current.pid)
    data_df['gcc_compile_result'] = data_df.apply(do_c_compile, raw=True, axis=1, file_path=file_path)
    return data_df


def transform_data(one, reverse_verdict, reverse_langdict):
    item = []
    item += [one['id']]
    item += [one['submit_url']]
    item += [one['problem_id']]
    item += [one['user_id']]
    item += [one['problem_user_id']]
    item += [reverse_langdict[one['language']]]
    item += [reverse_verdict[one['status']]]
    item += [one['error_test_id']]
    item += [1 if one['gcc_compile_result'] else 0]
    item += [one['code']]
    return item


def save_c_submit_code(data_df_list):
    create_table(COMPILE_SUCCESS_DATA_DBPATH, C_COMPILE_SUCCESS_RECORDS)

    result_list = [data_df['gcc_compile_result'].map(lambda x: 1 if x else 0) for data_df in data_df_list]
    count_list = [len(data_df) for data_df in data_df_list]
    success_res = np.sum(result_list)
    count_res = np.sum(count_list)
    print('success_res total: {}, total: {}'.format(success_res, count_res))

    def trans(error_df, reverse_verdict, reverse_langdict):
        res = [transform_data(row, reverse_verdict, reverse_langdict) for index, row in error_df.iterrows()]
        return res

    reverse_verdict = reverse_dict(verdict)
    reverse_langdict = reverse_dict(langdict)

    data_items_list = [trans(data_df, reverse_verdict, reverse_langdict) for data_df in data_df_list]
    for data_items in data_items_list:
        insert_items(COMPILE_SUCCESS_DATA_DBPATH, C_COMPILE_SUCCESS_RECORDS, data_items)



def filter_ac_c_records(data_df):
    data_df = data_df[data_df['status'].map(lambda x: x in [1, 3])]
    return data_df


def do_c_code_compile_main():
    data_df = read_all_c_records()
    print('all c records: {}'.format(len(data_df)))
    data_df = filter_ac_c_records(data_df)
    print('after filter ac c code: {}'.format(len(data_df)))
    data_df = init_c_code(data_df)
    print('after filter init c code: {}'.format(len(data_df)))

    data_df_list = []
    for g, df in data_df.groupby(np.arange(len(data_df)) // 100):
        data_df_list += [df]
    print('data_df_list length: ', len(data_df_list))
    sys.stdout.flush()
    sys.stderr.flush()

    chuck_count = 0
    for chu in chunks(data_df_list, 100):
        print('chuck {} start'.format(chuck_count))
        chuck_count += 1
        res = list(parallel_map(10, compile_map_fn, chu))
        save_c_submit_code(res)


if __name__ == '__main__':
    do_c_code_compile_main()









