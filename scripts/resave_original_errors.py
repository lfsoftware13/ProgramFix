import pandas as pd

from scripts.scripts_util import read_experiment_result_df, read_original_error_info, save_compile_result


def create_save_info(df:pd.DataFrame):
    save_records = []
    for row in df.itertuples():
        save_records += [(row.original_error_info, row.original_errors, row.original_error_count, row.id)]
    return save_records


def resave_original_errors_info_main(db_path, table_name):
    original_df = read_original_error_info()
    save_records = create_save_info(original_df)
    save_compile_result(save_records, db_path, table_name, command_key='update_original_compile_info')


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config4'
    resave_original_errors_info_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
