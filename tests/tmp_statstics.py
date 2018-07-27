import sqlite3
import pandas as pd

from common.pycparser_util import tokenize_by_clex_fn
from config import SLK_SAMPLE_DBPATH
from experiment.experiment_util import convert_c_code_fields_to_cpp_fields, action_list_sorted, load_common_error_data
from experiment.parse_xy_util import parse_error_tokens_and_action_map
from read_data.load_data_vocabulary import create_common_error_vocabulary
from read_data.read_data_from_db import read_data, read_slk_grammar_sample_train_records, \
    read_slk_grammar_sample_valid_records, read_slk_grammar_sample_test_records
from read_data.read_experiment_data import read_fake_common_c_error_dataset_with_limit_length


MAX_TOKEN_LENGTH = 500

count = 0
def update_column(df:pd.DataFrame, table_name, con):
    sql = '''update <TABLE> set original_modify_action_list=?, original_distance=? where id=?'''
    sql = sql.replace('<TABLE>', table_name)

    save_list = []
    for i, row in df.iterrows():
        global count
        count+= 1
        if count % 100 == 0:
            print(count)
        row_id = row['id']
        actions = row['modify_action_list_y']
        distance = row['distance_y']
        save_list += [(actions, distance, row_id)]

        if len(save_list)% 1000 == 0:
            con.executemany(sql, save_list)
            con.commit()
            save_list = []
    con.executemany(sql, save_list)
    con.commit()


def copy_distance_and_action_main():
    db_path = r'/home/lf/Project/ProgramFix/data/slk_sample_data.db'
    con = sqlite3.connect(db_path)

    sample_train = read_data(con, 'slk_sample_common_c_error_records_train')
    sample_valid = read_data(con, 'slk_sample_common_c_error_records_valid')
    sample_test = read_data(con, 'slk_sample_common_c_error_records_test')
    print(len(sample_train))
    print(len(sample_valid))
    print(len(sample_test))

    train, vaild, test = read_fake_common_c_error_dataset_with_limit_length(MAX_TOKEN_LENGTH)

    # train = train.sample(100)
    # vaild = vaild.sample(100)
    # test = test.sample(100)

    merge_train = pd.merge(sample_train, train, on=['id'])
    merge_valid = pd.merge(sample_valid, vaild, on=['id'])
    merge_test = pd.merge(sample_test, test, on=['id'])

    print(len(merge_train))
    print(len(merge_valid))
    print(len(merge_test))

    update_column(merge_train, 'slk_sample_common_c_error_records_train', con)
    update_column(merge_valid, 'slk_sample_common_c_error_records_valid', con)
    update_column(merge_test, 'slk_sample_common_c_error_records_test', con)


def statictics_distance_count(df: pd.DataFrame, df2: pd.DataFrame):

    def count_df_distance(df):
        df_len = len(df)
        count_df = df[['includes', 'distance']].groupby(['distance']).count()
        count_df['radio'] = count_df['includes'] / df_len
        return count_df

    count_df = count_df_distance(df)
    count_df2 = count_df_distance(df2)
    res = pd.merge(count_df, count_df2, how='outer', on='distance')
    return res


def distance_statistics_main():
    sample_train = read_slk_grammar_sample_train_records()
    sample_valid = read_slk_grammar_sample_valid_records()
    sample_test = read_slk_grammar_sample_test_records()
    total_len_train = len(sample_train)
    total_len_valid = len(sample_valid)
    total_len_test = len(sample_test)

    sample_train['improved'] = sample_train['original_distance'] - sample_train['distance']
    sample_valid['improved'] = sample_valid['original_distance'] - sample_valid['distance']
    sample_test['improved'] = sample_test['original_distance'] - sample_test['distance']

    def replace_error_distance(one):
        dis = one['distance']
        if dis == -1:
            one['improved'] = -100
        return one

    sample_train = sample_train.apply(replace_error_distance, raw=True, axis=1)
    sample_valid = sample_valid.apply(replace_error_distance, raw=True, axis=1)
    sample_test = sample_test.apply(replace_error_distance, raw=True, axis=1)

    sample_train_improved = sample_train[['id', 'improved']].groupby(['improved']).count()
    sample_valid_improved = sample_valid[['id', 'improved']].groupby(['improved']).count()
    sample_test_improved = sample_test[['id', 'improved']].groupby(['improved']).count()

    sample_train_improved['improved_radio'] = sample_train_improved['id'] / total_len_train
    sample_valid_improved['improved_radio'] = sample_valid_improved['id'] / total_len_valid
    sample_test_improved['improved_radio'] = sample_test_improved['id'] / total_len_test

    print(sample_train_improved)
    print(sample_valid_improved)
    print(sample_test_improved)

    def check_improved_label(x):
        if x == -100:
            return 'Error'
        if x == 0:
            return 'Same'
        if x < 0:
            return 'Not Improved'
        if x > 0:
            return 'Improved'
        return 'I dont know'

    sample_train_improved['class'] = sample_train_improved.index.map(check_improved_label)
    sample_valid_improved['class'] = sample_valid_improved.index.map(check_improved_label)
    sample_test_improved['class'] = sample_test_improved.index.map(check_improved_label)

    sample_train_improved_sum = sample_train_improved[['id', 'class', 'improved_radio']].groupby(['class']).sum()
    sample_valid_improved_sum = sample_valid_improved[['id', 'class', 'improved_radio']].groupby(['class']).sum()
    sample_test_improved_sum = sample_test_improved[['id', 'class', 'improved_radio']].groupby(['class']).sum()

    print(sample_train_improved_sum)
    print(sample_valid_improved_sum)
    print(sample_test_improved_sum)


    # print(len(sample_train))
    # print(len(sample_valid))
    # print(len(sample_test))

    # train_dict, valid_dict, test_dict = load_common_error_data()
    # train_df = pd.DataFrame(train_dict)
    # valid_df = pd.DataFrame(valid_dict)
    # test_df = pd.DataFrame(test_dict)
    # count_train = statictics_distance_count(sample_train, train_df)
    # count_valid = statictics_distance_count(sample_valid, valid_df)
    # count_test = statictics_distance_count(sample_test, test_df)
    # print(count_train)
    # print(count_valid)
    # print(count_test)


if __name__ == '__main__':
    distance_statistics_main()











