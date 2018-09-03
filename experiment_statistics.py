

config1 = {
    'result': '''
    correct_count: 2313/6975.0, correct: 0.33161290322580644
    part_correct_count: 1168/6975.0, part_correct: 0.1674551971326165
    model_total_error: 12348, original_total_error: 16766.0, resolved: 0.26350948347846836
    ''',
    'config_name': 'encoder_sample_config1',
    'encoder_type': 'rnn',
    'output_type': 'slice_action',
    'only_sample': False,
    'best_model': 'encoder_sample_dropout_overfitting.pkl',
    'run_result': 0.33638743455497383,
    'real_result': 0.33161290322580644,
    'table_name': 'encoder_sample_dropout_overfitting',
}


config2 = {
    'result': '''
    correct_count: 3469/6975.0, correct: 0.49734767025089605
    part_correct_count: 1093/6975.0, part_correct: 0.15670250896057347
    model_total_error: 9413, original_total_error: 16766.0, resolved: 0.4385661457712037
    ''',
    'config_name': 'encoder_sample_config2',
    'encoder_type': 'mixed',
    'output_type': 'slice_action',
    'only_sample': False,
    'best_model': 'graph_encoder_sample_config2.pkl24',
    'run_result': 0.5053904428904429,
    'real_result': 0.4973476702508960,
    'table_name': 'graph_encoder_sample_config2_24',
}


config3 = {
    'result': '''
    correct_count: 1272/6975.0, correct: 0.18236559139784947
    part_correct_count: 963/6975.0, part_correct: 0.13806451612903226
    model_total_error: 23244, original_total_error: 16766.0, resolved: -0.386377191936061
    ''',
    'config_name': 'encoder_sample_config3',
    'encoder_type': 'ggnn',
    'output_type': 'slice_action',
    'only_sample': False,
    'best_model': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link.pkl79',
    'run_result': 0.1849912739965096,
    'real_result': 0.18236559139784947,
    'table_name': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link_79',
}


config4 = {
    'result': '''
    correct_count: 3834/6975.0, correct: 0.5496774193548387
    part_correct_count: 1019/6975.0, part_correct: 0.1460931899641577
    model_total_error: 9351, original_total_error: 16766.0, resolved: 0.44226410592866516
    ''',
    'config_name': 'encoder_sample_config4',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': False,
    'best_model': 'encoder_sample_config4.pkl20',
    'run_result': 0.5585664335664335,
    'real_result': 0.5496774193548387,
    'table_name': 'encoder_sample_config4_20',
}


config5 = {
    'result': '''
    correct_count: 3032/6975.0, correct: 0.4346953405017921
    part_correct_count: 974/6975.0, part_correct: 0.1396415770609319
    model_total_error: 12138, original_total_error: 16766.0, resolved: 0.2760348323989026
    ''',
    'config_name': 'encoder_sample_config5',
    'encoder_type': 'mixed',
    'output_type': 'slice_action',
    'only_sample': True,
    'best_model': 'encoder_sample_config5.pkl11',
    'run_result': 0.4417249417249417,
    'real_result': 0.4346953405017921,
    'table_name': 'encoder_sample_config5_11',
}


config6 = {
    'result': '''
    correct_count: 2511/6975.0, correct: 0.36
    part_correct_count: 1205/6975.0, part_correct: 0.17275985663082438
    model_total_error: 11610, original_total_error: 16766.0, resolved: 0.3075271382559943
    ''',
    'config_name': 'encoder_sample_config6',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': False,
    'best_model': 'encoder_sample_config6_only_gru_with_token_action.pkl19',
    'run_result': 0.36582167832167833,
    'real_result': 0.36,
    'table_name': 'encoder_sample_config6_only_gru_with_token_action_19',
}


config7 = {
    # may be not overfitting
    'result': '''
    correct_count: 2262/6975.0, correct: 0.3243010752688172
    part_correct_count: 731/6975.0, part_correct: 0.10480286738351255
    model_total_error: 19024, original_total_error: 16766.0, resolved: -0.13467732315400216
    ''',
    'config_name': 'encoder_sample_config7',
    'encoder_type': 'ggnn',
    'output_type': 'token_action',
    'only_sample': False,
    'best_model': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link.pkl25',
    'run_result': 0.32954545454545453,
    'real_result': 0.3243010752688172,
    'table_name': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link_25',
}


config8 = {
    'result': '''
    correct_count: 3470/6975.0, correct: 0.4974910394265233
    part_correct_count: 1066/6975.0, part_correct: 0.15283154121863798
    model_total_error: 9843, original_total_error: 16766.0, resolved: 0.4129190027436479
    ''',
    'config_name': 'encoder_sample_config8',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': True,
    'best_model': 'encoder_sample_config8.pkl15',
    'run_result': 0.5055361305361306,
    'real_result': 0.4974910394265233,
    'table_name': 'encoder_sample_config8_15',
}


config9 = {
    'result': '''
    correct_count: 2543/6975.0, correct: 0.3645878136200717
    part_correct_count: 1043/6975.0, part_correct: 0.14953405017921148
    model_total_error: 13351, original_total_error: 16766.0, resolved: 0.20368603125372775
    ''',
    'config_name': 'encoder_sample_config9',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': True,
    'best_model': 'encoder_sample_config9_only_gru_with_token_action_only_sample.pkl10',
    'run_result': 0.370483682983683,
    'real_result': 0.3645878136200717,
    'table_name': 'encoder_sample_config9_only_gru_with_token_action_only_sample_10',
}


config10 = {
    # no finish
    'result': '',
    'config_name': 'encoder_sample_config10',
    'encoder_type': 'ggnn',
    'output_type': 'token_action',
    'only_sample': True,
    'best_model': '',
    'run_result': 0,
    'real_result': 0,
    'table_name': '',
}


config11 = {
    'result': '''
    correct_count: 3974/6975.0, correct: 0.5697491039426523
    part_correct_count: 947/6975.0, part_correct: 0.13577060931899643
    model_total_error: 11119, original_total_error: 16766.0, resolved: 0.3368125969223429
    ''',
    'config_name': 'encoder_sample_config11',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config11.pkl23',
    'run_result': 0.578962703962704,
    'real_result': 0.5697491039426523,
    'table_name': 'encoder_sample_config11_23',


    'one_step_result': '''
    correct_count: 2163/6975.0, correct: 0.31010752688172044
    part_correct_count: 1880/6975.0, part_correct: 0.26953405017921145
    model_total_error: 12170, original_total_error: 16766.0, resolved: 0.2741262078015031
    ''',
    'one_step_run_result': 0.3151223776223776,
    'one_step_real_result': 0.31010752688172044,

}


config12 = {
    'result': '''
    correct_count: 2384/6975.0, correct: 0.3417921146953405
    part_correct_count: 861/6975.0, part_correct: 0.12344086021505377
    model_total_error: 16654, original_total_error: 16766.0, resolved: 0.006680186090898266
    ''',
    'config_name': 'encoder_sample_config12',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config12_only_gru_with_token_action.pkl11',
    'run_result': 0.3473193473193473,
    'real_result': 0.3417921146953405,
    'table_name': 'encoder_sample_config12_only_gru_with_token_action_11',
}


config13 = {
    'result': '''
    correct_count: 2092/6975.0, correct: 0.2999283154121864
    part_correct_count: 724/6975.0, part_correct: 0.10379928315412186
    model_total_error: 21516, original_total_error: 16766.0, resolved: -0.2833114636764882
    ''',
    'config_name': 'encoder_sample_config13',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link.pkl22',
    'run_result': 0.30495626822157434,
    'real_result': 0.2999283154121864,
    'table_name': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link_22',
}


config14 = {
    'result': '''
    correct_count: 3312/6975.0, correct: 0.47483870967741937
    part_correct_count: 1013/6975.0, part_correct: 0.14523297491039427
    model_total_error: 10344, original_total_error: 16766.0, resolved: 0.383037098890612
    ''',
    'config_name': 'encoder_sample_config14',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': True,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config14.pkl20',
    'run_result': 0.4825174825174825,
    'real_result': 0.47483870967741937,
    'table_name': 'encoder_sample_config14_20',
}


config15 = {
    'result': '''
    correct_count: 2623/6975.0, correct: 0.3760573476702509
    part_correct_count: 1550/6975.0, part_correct: 0.2222222222222222
    model_total_error: 10783, original_total_error: 16766.0, resolved: 0.3568531551950376
    ''',
    'config_name': 'encoder_sample_config15',
    'encoder_type': 'mixed',
    'output_type': 'sequence',
    'only_sample': False,
    'best_model': 'encoder_sample_config15_sequence_output.pkl63',
    'run_result': 0.38213869463869465,
    'real_result': 0.3760573476702509,
    'table_name': 'encoder_sample_config15_sequence_output_63',
}


config16 = {
    'result': '''
    correct_count: 2561/6975.0, correct: 0.367168458781362
    part_correct_count: 1071/6975.0, part_correct: 0.15354838709677418
    model_total_error: 12500, original_total_error: 16766.0, resolved: 0.25444351664082066
    ''',
    'config_name': 'encoder_sample_config16',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': True,
    'decoder_type': 'fnn',
    'best_model': 'encoder_sample_config16_only_gru_with_token_action_only_sample_14.pkl14',
    'run_result': 0.3731060606060606,
    'real_result': 0.367168458781362,
    'table_name': 'encoder_sample_config16_only_gru_with_token_action_only_sample_14',
}


sensibility_config_2 = {
    'result': '''
    correct_count: 60/6975.0, correct: 0.008602150537634409
    part_correct_count: 937/6975.0, part_correct: 0.134336917562724
    model_total_error: 26643, original_total_error: 16766.0, resolved: -0.5891089108910892
    ''',
    'config_name': 'sensibility_rnn_config2',
    'best_model': 'sensibility_rnn_config2.pkl81',
    'run_result': 0,
    'real_result': 0.008602150537634409,
    'table_name': 'sensibility_rnn_config2_81_filter',
}
