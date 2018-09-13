from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from scripts.scripts_util import read_experiment_result_df


def step_bar_figure(c):
    ind = 1 + np.arange(9)
    steps_value = [c[i] for i in ind]

    f = plt.figure(figsize=(4, 3))
    p1 = plt.bar(ind, steps_value, 0.35)

    plt.ylabel('Count of Code')
    plt.xlabel('Sample Steps')
    plt.xticks(ind, [str(i) for i in ind])

    plt.show()
    f.savefig('sample_steps_count.png', bbox_inches='tight', dpi=300)


def draw_sample_step_main(db_path, table_name):
    df = read_experiment_result_df(db_path, table_name)
    df = df[df['compile_res'].map(lambda x: x == 1)]
    print(len(df))
    sample_step_list = df['sample_step'].tolist()
    c = Counter(sample_step_list)
    print(c)
    step_bar_figure(c)




if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config11_23'
    draw_sample_step_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)