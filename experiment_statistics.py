

config1 = {
    'config_name': 'encoder_sample_config1',
    'encoder_type': 'rnn',
    'output_type': 'slice_action',
    'only_sample': False,
    'best_model': 'encoder_sample_dropout_overfitting.pkl',
    'run_result': 0.33638743455497383,
    'real_result': 0,
    'table_name': 'encoder_sample_dropout_overfitting',
}

config2 = {
    'config_name': 'encoder_sample_config2',
    'encoder_type': 'mixed',
    'output_type': 'slice_action',
    'only_sample': False,
    'best_model': 'graph_encoder_sample_config2.pkl24',
    'run_result': 0.5053904428904429,
    'real_result': 0,
    'table_name': 'graph_encoder_sample_config2_24',
}


config3 = {
    'config_name': 'encoder_sample_config3',
    'encoder_type': 'ggnn',
    'output_type': 'slice_action',
    'only_sample': False,
    'best_model': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link.pkl79',
    'run_result': 0.1849912739965096,
    'real_result': 0,
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
    'real_result': 0,
    'table_name': 'encoder_sample_config4_20',
}


config5 = {
    'config_name': 'encoder_sample_config5',
    'encoder_type': 'mixed',
    'output_type': 'slice_action',
    'only_sample': True,
    'best_model': 'encoder_sample_config5.pkl11',
    'run_result': 0.4417249417249417,
    'real_result': 0,
    'table_name': 'encoder_sample_config5_11',
}


config6 = {
    'config_name': 'encoder_sample_config6',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': False,
    'best_model': 'encoder_sample_config6_only_gru_with_token_action.pkl19',
    'run_result': 0.36582167832167833,
    'real_result': 0,
    'table_name': 'encoder_sample_config6_only_gru_with_token_action_19',
}


config7 = {
    # may be not overfitting
    'config_name': 'encoder_sample_config7',
    'encoder_type': 'ggnn',
    'output_type': 'token_action',
    'only_sample': False,
    'best_model': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link.pkl25',
    'run_result': 0.32954545454545453,
    'real_result': 0,
    'table_name': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link_25',
}


config8 = {
    'config_name': 'encoder_sample_config8',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': True,
    'best_model': 'encoder_sample_config8.pkl13',
    'run_result': 0.4985431235431235,
    'real_result': 0,
    'table_name': 'encoder_sample_config8_13',
}


config9 = {
    'config_name': 'encoder_sample_config9',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': True,
    'best_model': 'encoder_sample_config9_only_gru_with_token_action_only_sample.pkl',
    'run_result': 0,
    'real_result': 0,
    'table_name': 'encoder_sample_config9_only_gru_with_token_action_only_sample',
}