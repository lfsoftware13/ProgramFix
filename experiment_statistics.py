

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