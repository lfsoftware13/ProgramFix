import sqlite3
import pandas as pd

from common.constants import DATA_RECORDS_DEEPFIX
from database.database_util import run_sql_statment
from read_data.read_data_from_db import read_data, read_deepfix_records


def read_experiment_result_df(db_path, table_name) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = read_data(conn, table_name)
    return df


def read_deepfix_error_records() -> pd.DataFrame:
    df = read_deepfix_records()
    df = df[df['errorcount'].map(lambda x: x > 0)]
    df['id'] = df['code_id']
    return df


def read_original_error_info() -> pd.DataFrame:
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    db_path = DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'original_compile_info'
    conn = sqlite3.connect(db_path)
    df = read_data(conn, table_name)
    return df


def save_compile_result(save_records, db_path, table_name, command_key):
    if len(save_records) != 0:
        run_sql_statment(db_path, DATA_RECORDS_DEEPFIX, command_key, save_records, replace_table_name=table_name)


if __name__ == '__main__':
    df = read_deepfix_error_records()
    total_error = sum(df['errorcount'].tolist())
    print(total_error)
