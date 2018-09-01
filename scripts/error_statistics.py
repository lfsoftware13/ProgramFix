import itertools
import json
import re
from collections import Counter

import more_itertools
from sklearn.utils import shuffle

from scripts.scripts_util import read_experiment_result_df, read_deepfix_error_records


def replace_itentifier(msg):
    msg = re.sub("'.*?'", '<CODE>', msg)
    return msg


def stat_error_type_count(error_list):
    c = Counter(error_list)
    return c


def calculate_top_error_type(db_path, table_name, key):
    df = read_experiment_result_df(db_path, table_name)
    df = df[df[key].map(lambda x: x is not '')]
    df[key] = df[key].map(json.loads)
    standard_key = 'standard_'+key
    df[standard_key] = df[key].map(
        lambda x: [replace_itentifier(i) for i in x])
    # df['standard_original_errors_set'] = df['standard_original_errors'].map(set)
    standard_error_list = set(more_itertools.collapse(
        df[standard_key].tolist()))
    standard_dict = {e: 0 for e in standard_error_list}

    for e_list in df[standard_key]:
        e_set = set(e_list)
        for e in e_set:
            standard_dict[e] += 1

    # standard_error_list = [replace_itentifier(e) for e in error_list]
    # c = stat_error_type_count(standard_error_list)
    return standard_dict


def calculate_error_msg_count(db_path, table_name, key):
    df = read_experiment_result_df(db_path, table_name)
    df = df[df[key].map(lambda x: x is not '')]
    df[key] = df[key].map(json.loads)
    error_list = list(more_itertools.collapse(df[key].tolist()))
    standard_error_list = [replace_itentifier(e) for e in error_list]
    c = stat_error_type_count(standard_error_list)
    return c


def calculate_error_type_main2(db_path, table_name):
    after_top_dict = calculate_top_error_type(db_path, table_name, 'errors')
    top_dict = calculate_top_error_type(db_path, table_name, 'original_errors')

    sorted_top_dict = list(sorted(top_dict.items(), key=lambda kv: kv[1], reverse=True))
    # sorted_after_top_dict = list(sorted(after_top_dict.items(), key=lambda kv: kv[1], reverse=True))

    print(len(top_dict), len(after_top_dict))

    m = 0
    for k, v in sorted_top_dict:
        m += 1
        if m > 10:
            break
        print('{}: {}({}) \ {}'.format(k, after_top_dict[k], v-after_top_dict[k], v))


def calculate_error_type_main(db_path, table_name):
    top_c = calculate_error_msg_count(db_path, table_name, 'original_errors')
    after_c = calculate_error_msg_count(db_path, table_name, 'errors')

    m = 0
    print('errors in {}'.format(table_name))
    sorted_top_c = list(sorted(top_c.items(), key=lambda kv: kv[1], reverse=True))
    for k, v in sorted_top_c:
        print('{}: {}({}) \ {}'.format(k, after_c[k], v-after_c[k], v))
        # p = 0
        # for e, s in zip(*shuffle(error_list, standard_error_list)):
        #     if s == k:
        #         print('    '+e)
        #         p += 1
        #         if p >= 10:
        #             break
        m += 1
        if m > 10:
            break


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config4_20'
    # calculate_error_type_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    calculate_error_type_main2(DATA_RECORDS_DEEPFIX_DBPATH, table_name)