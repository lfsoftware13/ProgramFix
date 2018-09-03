from common.constants import DATA_RECORDS_DEEPFIX
from database.database_util import create_table, run_sql_statment
from scripts.scripts_util import read_experiment_result_df

import pandas as pd


def select_best_records(df: pd.DataFrame):
    one = None
    for row in df.itertuples():
        one = (row.id, row.includes, row.code, row.sample_code, row.code_list, row.sample_code_list, row.compile_res, row.sample_step, row.sample_records)
        if row.sample_step > 0 and row.compile_res == 1:
            return one
    return one


def filter_program_id_main(db_path, table_name, new_table_name):
    df = read_experiment_result_df(db_path, table_name)
    grouped = df.groupby('id')
    print('group length: ', len(grouped))
    save_list = []
    for name, group in grouped:
        one = select_best_records(group)
        save_list += [one]

    print('save list length: ', len(save_list))
    create_table(db_path, DATA_RECORDS_DEEPFIX, replace_table_name=new_table_name)
    run_sql_statment(db_path, DATA_RECORDS_DEEPFIX, 'insert_ignore', save_list, replace_table_name=new_table_name)


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'sensibility_rnn_config2_81'
    new_table_name = 'sensibility_rnn_config2_81_filter'
    filter_program_id_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name, new_table_name)
