import json
import re
from collections import Counter

import more_itertools

from scripts.scripts_util import read_experiment_result_df


def replace_itentifier(msg):
    msg = re.sub("'.*?'", '<CODE>', msg)
    return msg


def stat_error_type_count(error_list):
    c = Counter(error_list)
    return c


def calculate_error_type_main(db_path, table_name, key):
    df = read_experiment_result_df(db_path, table_name)
    df = df[df[key].map(lambda x: x is not '')]
    df[key] = df[key].map(json.loads)
    error_list = list(more_itertools.collapse(df[key].tolist()))
    standard_error_list = [replace_itentifier(e) for e in error_list]
    c = stat_error_type_count(standard_error_list)

    m = 0
    print('errors in {}[{}]'.format(table_name, key))
    sorted_by_value = sorted(c.items(), key=lambda kv: kv[1], reverse=True)
    for k, v in sorted_by_value:
        print('{}: {}'.format(k, v))
        for e, s in zip(error_list, standard_error_list):
            if s == k:
                print(e)
                break
        m += 1
        if m > 9:
            break


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config4_20'
    calculate_error_type_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name, key='original_errors')
