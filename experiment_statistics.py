

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
    correct_count: 4054/6975.0, correct: 0.5812186379928316
    part_correct_count: 1001/6975.0, part_correct: 0.14351254480286738
    model_total_error: 8602, original_total_error: 16766.0, resolved: 0.48693785041154714''',
    'config_name': 'encoder_sample_config11',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config11.pkl25',
    'run_result': 0.5901878002620469,
    'real_result': 0.5812186379928316,
    'table_name': 'encoder_sample_config11_25',


    'one_step_result': '''
    correct_count: 2164/6975.0, correct: 0.3102508960573477
    part_correct_count: 1866/6975.0, part_correct: 0.2675268817204301
    model_total_error: 11922, original_total_error: 16766.0, resolved: 0.28891804843134916''',
    'one_step_run_result': 0.3150385791235988,
    'one_step_real_result': 0.3102508960573477,

}


config12 = {
    'result': '''
    correct_count: 2549/6975.0, correct: 0.36544802867383513
    part_correct_count: 1109/6975.0, part_correct: 0.15899641577060933
    model_total_error: 12844, original_total_error: 16766.0, resolved: 0.23392580221877612''',
    'config_name': 'encoder_sample_config12',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config12_only_gru_with_token_action.pkl27',
    'run_result': 0.37108749454069007,
    'real_result': 0.36544802867383513,
    'table_name': 'encoder_sample_config12_only_gru_with_token_action_27',
}


config13 = {
    'result': '''
    correct_count: 2101/6975.0, correct: 0.30121863799283155
    part_correct_count: 648/6975.0, part_correct: 0.09290322580645162
    model_total_error: 22647, original_total_error: 16766.0, resolved: -0.35076941429082664''',
    'config_name': 'encoder_sample_config13',
    'encoder_type': 'ggnn',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link.pkl21',
    'run_result': 0.3058669384189838,
    'real_result': 0.30121863799283155,
    'table_name': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link_21',
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
    part_correct_count: 995/6975.0, part_correct: 0.14265232974910394
    model_total_error: 13328, original_total_error: 16766.0, resolved: 0.2050578551831087''',
    'config_name': 'encoder_sample_config16',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': True,
    'decoder_type': 'fnn',
    'best_model': 'encoder_sample_config16_only_gru_with_token_action_only_sample.pkl12',
    'run_result': 0.3728344737225215,
    'real_result': 0.367168458781362,
    'table_name': 'encoder_sample_config16_only_gru_with_token_action_only_sample_12',
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


config18 = {
    'result': '''
    correct_count: 3090/6975.0, correct: 0.443010752688172
    part_correct_count: 804/6975.0, part_correct: 0.11526881720430107
    model_total_error: 14652, original_total_error: 16766.0, resolved: 0.1260885124657044''',
    'config_name': 'encoder_sample_config18',
    'dataset': 'codeforces',
    'encoder_type': 'mixed',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': 'encoder_sample_config18.pkl24',
    'run_result': 0.44984713932158976,
    'real_result': 0.443010752688172,
    'table_name': 'encoder_sample_config18_24',

}


config19 = {
    'result': '''''',
    'config_name': 'encoder_sample_config19',
    'dataset': 'codeforces',
    'encoder_type': 'rnn',
    'output_type': 'token_action',
    'only_sample': False,
    'decoder_type': 'ffn',
    'best_model': '',
    'run_result': 0,
    'real_result': 0,
    'table_name': '',
}
