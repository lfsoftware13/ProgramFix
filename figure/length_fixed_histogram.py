import json
import matplotlib.pyplot as plt
import pandas as pd

from scripts.scripts_util import read_experiment_result_df


def length_statistics(db_path, table_name):
    df = read_experiment_result_df(db_path, table_name)
    df = df[df['code_list'].map(lambda x: x != '')]
    df['code_list'] = df['code_list'].map(json.loads)
    df['code_length'] = df['code_list'].map(len)
    df['part_result'] = df.apply(lambda one: 1 if one['compile_res'] != 1 and
                                                  one['error_count'] < one['original_error_count'] and
                                                  one['error_count'] > 0 else 0, raw=True, axis=1)
    code_length = df['code_length'].tolist()
    compile_res = df['compile_res'].tolist()
    part_res = df['part_result'].tolist()
    print('min length: {}, max_length: {}'.format(min(code_length), max(code_length)))
    return code_length, compile_res, part_res


def length_figure(code_length, compile_res, part_compile_res):
    true_length = [l if r == 1 else None for l, r, in zip(code_length, compile_res)]
    # part_true_length = [l if r == 1 else None for l, r, in zip(code_length, part_compile_res)]
    error_length = [l if r == 0 else None for l, r in zip(code_length, compile_res)]
    true_length = list(filter(lambda x: x is not None, true_length))
    # part_true_length = list(filter(lambda x: x is not None, part_true_length))
    error_length = list(filter(lambda x: x is not None, error_length))
    print(len(code_length), len(true_length))

    f = plt.figure(figsize=(2, 1.5))
    n, bins, patches = plt.hist([true_length, error_length], 26, stacked=True, ec='gray', label=['Fixed', 'Unfixed'])
    print(sum(n[0]), sum(n[1]))
    plt.xlabel('Code Length')
    plt.ylabel('Count of Code')
    plt.legend()
    plt.show()
    f.savefig('code_length_histogram_0.5.pdf', bbox_inches='tight', dpi=300)


def length_main(db_path, table_name):
    code_length, compile_res, part_res = length_statistics(db_path, table_name)
    length_figure(code_length, compile_res, part_res)


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config11_23'
    length_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)