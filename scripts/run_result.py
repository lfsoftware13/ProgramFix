from scripts.insert_unsample_deepfix_records import main_insert_unsample_records_to_database
from scripts.resave_original_errors import resave_original_errors_info_main
from scripts.save_error_info import check_error_count_main, main_compile_code_and_read_error_info
from scripts.stat_error import stat_main

if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    # table_name = 'encoder_sample_dropout_overfitting'
    # table_name = 'graph_encoder_sample_config2_24'
    # table_name = 'graph_encoder_sample_config3_only_ggnn_with_sequence_link_79'
    # table_name = 'graph_encoder_sample_config3_only_ggnn_with_sequence_link_24'
    # table_name = 'encoder_sample_config4_20'
    # table_name = 'encoder_sample_config5_11'
    # table_name = 'encoder_sample_config6_only_gru_with_token_action_19'
    # table_name = 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link_25'
    # table_name = 'encoder_sample_config8_15'
    # table_name = 'encoder_sample_config9_only_gru_with_token_action_only_sample_10'
    # table_name = 'encoder_sample_config11_23'
    # table_name = 'encoder_sample_config12_only_gru_with_token_action_11'
    table_name = 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link_22'
    # table_name = 'encoder_sample_config14_11'
    # table_name = 'encoder_sample_config14_20'
    # table_name = 'encoder_sample_config16_only_gru_with_token_action_only_sample_14'
    # table_name = 'sensibility_rnn_config2_81_filter'
    # table_name = 'encoder_sample_config15_sequence_output_63'

    main_insert_unsample_records_to_database(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    resave_original_errors_info_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    check_error_count_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)
    main_compile_code_and_read_error_info(DATA_RECORDS_DEEPFIX_DBPATH, table_name,
                                          do_compile_original=False)
    stat_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name, compile_result=True,
              part_correct=True, error_solver=True)
