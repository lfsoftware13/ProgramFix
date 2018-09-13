import json
import matplotlib.pyplot as plt
import numpy as np

from scripts.scripts_util import read_experiment_result_df


def draw_line_figure(bin_list, name_list):
    bin_list = [list(sorted(bin.items(), key=lambda kv: kv[0])) for bin in bin_list]
    items_list = [[(transform_bin_length(k), v) for k, v in bin] for bin in bin_list]
    data_list = [list(zip(*items)) for items in items_list]

    f = plt.figure(figsize=(4, 3))
    plt.plot(data_list[0][0], data_list[0][1], linewidth=2.0, label=name_list[0], linestyle="-")
    plt.plot(data_list[1][0], data_list[1][1], linewidth=2.0, label=name_list[1], linestyle="--")
    plt.plot(data_list[2][0], data_list[2][1], linewidth=2.0, label=name_list[2], linestyle=":")
    plt.xlabel('Code Length')
    plt.ylabel('Metrics')
    plt.yticks(np.linspace(0, 1, 6))
    plt.legend(fontsize=8, )
    plt.show()
    f.savefig('code_score_line_figure.png', bbox_inches='tight', dpi=300)


def transform_bin_id(length):
    bin_id = int((length-60)/20)
    return bin_id


def transform_bin_length(bin_id):
    l = bin_id * 20 + 70
    return l


def calculate_bin_count(result_list, length_list):
    bin = {}
    for res, l in zip(result_list, length_list):
        bin_id = transform_bin_id(l)
        bin[bin_id] = bin.get(bin_id, 0) + res
    return bin


def statistics_score(db_path, table_name):
    df = read_experiment_result_df(db_path, table_name)
    df = df[df['code_list'].map(lambda x: x != '')]
    df['code_list'] = df['code_list'].map(json.loads)
    df['code_length'] = df['code_list'].map(len)
    df['part_result'] = df.apply(lambda one: 1 if one['compile_res'] != 1 and
                                                  one['error_count'] < one['original_error_count'] and
                                                  one['error_count'] > 0 else 0, raw=True, axis=1)

    em_df = df[df['compile_res'].map(lambda x: x == 1)]
    pm_df = df[df['part_result'].map(lambda x: x == 1)]

    total_bin = calculate_bin_count([1 for _ in df['code_length'].tolist()], df['code_length'].tolist())
    em_bin = calculate_bin_count(em_df['compile_res'].tolist(), em_df['code_length'].tolist())
    pm_bin = calculate_bin_count(pm_df['part_result'].tolist(), pm_df['code_length'].tolist())
    print(total_bin)
    print(em_bin)
    print(pm_bin)

    em_ratio_bin = {k: em_bin[k]/v for k, v in total_bin.items()}
    pm_ratio_bin = {k: pm_bin.get(k, 0)/v for k, v in total_bin.items()}
    print(em_ratio_bin)
    print(pm_ratio_bin)

    error_msg_bin = calculate_bin_count(df['error_count'].tolist(), df['code_length'].tolist())
    total_error_msg_bin = calculate_bin_count(df['original_error_count'].tolist(), df['code_length'].tolist())
    print(error_msg_bin)
    print(total_error_msg_bin)

    resolved_ratio_bin = {k: (v - error_msg_bin[k]) / v for k, v in total_error_msg_bin.items()}
    print(resolved_ratio_bin)

    draw_line_figure([em_ratio_bin, pm_ratio_bin, resolved_ratio_bin], ['EM', 'PM', 'EMR'])




if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config11_23'
    statistics_score(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
