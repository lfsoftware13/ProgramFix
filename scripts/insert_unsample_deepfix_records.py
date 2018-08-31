import json

from common.analyse_include_util import extract_include, replace_include_with_blank
from common.constants import DATA_RECORDS_DEEPFIX
from database.database_util import create_table, run_sql_statment
from scripts.scripts_util import read_experiment_result_df, read_deepfix_error_records


def create_unsample_result_records(unsample_df):
    unsample_df['includes'] = unsample_df['code'].map(lambda x: extract_include(x)).copy()
    unsample_df['code'] = unsample_df['code'].map(lambda x: replace_include_with_blank(x)).copy()

    def create_one_records(one):
        # (id, includes, code, sample_code, code_list, sample_code_list, compile_res, sample_step, sample_records)
        rec = (one['id'], json.dumps(one['includes']), one['code'], '', '', '', 0, -1, '', )
        return rec

    unsample_records = unsample_df.apply(create_one_records, axis=1, raw=True).tolist()
    return unsample_records


def main_insert_unsample_records_to_database(db_path, table_name):
    deepfix_df = read_deepfix_error_records()
    result_df = read_experiment_result_df(db_path, table_name)
    exist_ids = result_df['id'].tolist()
    unsample_records_df = deepfix_df[deepfix_df['id'].map(lambda x: x not in exist_ids)]
    if len(unsample_records_df) != 0:
        unsample_records = create_unsample_result_records(unsample_records_df)

        create_table(db_path, DATA_RECORDS_DEEPFIX, replace_table_name=table_name)
        run_sql_statment(db_path, DATA_RECORDS_DEEPFIX, 'insert_ignore', unsample_records, replace_table_name=table_name)


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config4_20'
    main_insert_unsample_records_to_database(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
