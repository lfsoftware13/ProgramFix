from scripts.insert_unsample_deepfix_records import main_insert_unsample_records_to_database
from scripts.resave_original_errors import resave_original_errors_info_main
from scripts.save_error_info import check_error_count_main, main_compile_code_and_read_error_info
from scripts.stat_error import stat_main

if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'encoder_sample_config4_20'

    main_insert_unsample_records_to_database(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    resave_original_errors_info_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    check_error_count_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    main_compile_code_and_read_error_info(DATA_RECORDS_DEEPFIX_DBPATH, table_name, do_compile_original=True)
    stat_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name, compile_result=True, part_correct=True, error_solver=True)
