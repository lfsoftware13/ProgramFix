import itertools
import json
import re

from common.args_util import get_compile_pool
from common.constants import DATA_RECORDS_DEEPFIX
from common.util import compile_c_code_by_gcc, add_pid_to_file_path
from database.database_util import run_sql_statment
from scripts.scripts_util import read_experiment_result_df, read_deepfix_error_records, save_compile_result


def compile_and_read_error_info(code):
    file_path = '/dev/shm/main.c'
    target_file_path = '/dev/shm/main.out'
    log_file_path = '/dev/shm/main.log'
    res = compile_c_code_by_gcc(code, file_path=file_path, target_file_path=target_file_path,
                                log_file_path=log_file_path, add_pid=True)
    pid_log_file_path = add_pid_to_file_path(log_file_path)
    with open(pid_log_file_path, encoding='utf-8') as f:
        texts = f.read()
        texts = texts.replace(u"\u2018", "'").replace(u"\u2019", "'")
    return res, texts


def consist_full_code(one, code_key):
    code = one[code_key]
    includes = one['includes']
    full_code = '\n'.join(includes) + '\n' + code
    return full_code


def create_compile_info_save_records(ids, compile_info_list, error_list):
    save_records = []
    for i, info, errors in zip(ids, compile_info_list, error_list):
        save_records += [(info, json.dumps(errors), len(errors), i)]
    return save_records


def extract_error_lines(l):
    pattern = re.compile(r"^(.+\.c:)?\d+:\d+: error: (.*)$")
    match = pattern.search(l)
    if match:
        message = match.group(2)
        return message
    return None


def extract_error_message(info):
    info_lines = info.split('\n')
    info_res = [extract_error_lines(l) for l in info_lines]
    error_l = list(filter(lambda x: x is not None, info_res))
    return error_l


def pool_compile_and_save(full_original_code):
    compile_pool = get_compile_pool()
    compile_result_list = list(compile_pool.starmap(compile_and_read_error_info, [[i] for i in full_original_code]))
    # compile_result_list = list(itertools.starmap(compile_and_read_error_info, [[i] for i in full_original_code]))
    compile_res_list, compile_info_list = list(zip(*compile_result_list))
    error_list = [extract_error_message(info) for info in compile_info_list]
    return compile_res_list, compile_info_list, error_list


def main_compile_code_and_read_error_info(db_path, table_name, do_compile_original=False):
    df = read_experiment_result_df(db_path, table_name)
    df['includes'] = df['includes'].map(json.loads)
    if do_compile_original:
        df['full_code'] = df.apply(consist_full_code, raw=True, axis=1, code_key='code')
        ids = df['id'].tolist()
        full_original_code = df['full_code'].tolist()
        compile_res_list, compile_info_list, error_list = pool_compile_and_save(full_original_code)
        save_records = create_compile_info_save_records(ids, compile_info_list, error_list)
        save_compile_result(save_records, db_path, table_name, command_key='update_original_compile_info')

    # effect_df = df[df['sample_step'].map(lambda x: x >= 0)]
    # effect_df['full_sample_code'] = effect_df.apply(consist_full_code, raw=True, axis=1, code_key='sample_code')
    # effect_ids = effect_df['id'].tolist()
    # full_sample_code = effect_df['full_sample_code'].tolist()
    # compile_res_list, compile_info_list, error_list = pool_compile_and_save(full_sample_code)
    # save_records = create_compile_info_save_records(effect_ids, compile_info_list, error_list)
    # save_compile_result(save_records, db_path, table_name, command_key='update_sample_compile_info')


def check_error_count_main(db_path, table_name):
    deepfix_df = read_deepfix_error_records()
    ids = deepfix_df['id'].tolist()
    deepfix_error_count = deepfix_df['errorcount'].tolist()
    error_dict = {i: c for i, c in zip(ids, deepfix_error_count)}
    df = read_experiment_result_df(db_path, table_name)
    df_ids = df['id'].tolist()
    df_error_count = df['original_error_count'].tolist()
    df_error_dict = {i: c for i, c in zip(df_ids, df_error_count)}

    tot = 0
    for i in ids:
        if error_dict[i] != df_error_dict[i]:
            tot += 1
    print(tot)


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config4_20'
    main_compile_code_and_read_error_info(DATA_RECORDS_DEEPFIX_DBPATH, table_name, do_compile_original=True)
    # check_error_count_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
