from common.constants import pre_defined_c_tokens, pre_defined_c_library_tokens, \
    SLK_SAMPLE_COMMON_C_ERROR_RECORDS_BASENAME
from config import SLK_SAMPLE_DBPATH, DATA_RECORDS_DEEPFIX_DBPATH, save_model_root, \
    DATA_RECORDS_DEEPFIX_CODEFORCES_TRAIN_DBPATH
from common.evaluate_util import SLKOutputAccuracyAndCorrect, EncoderCopyAccuracyAndCorrect, \
    ErrorPositionAndValueAccuracy, SensibilityRNNEvaluator
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from model.encoder_sample_model import create_parse_input_batch_data_fn, create_records_all_output, \
    change_output_records_to_batch, create_save_database_records, sensibility_create_save_database_records, \
    change_output_records_to_batch_fn
from model.one_pointer_copy_self_attention_seq2seq_model_gammar_mask_refactor import load_sample_save_dataset, \
    create_save_sample_data
from read_data.load_data_vocabulary import create_common_error_vocabulary, create_deepfix_common_error_vocabulary
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK

import pandas as pd


def load_vocabulary():
    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                                                addition_tokens=addition_tokens)
    special_token = pre_defined_c_tokens | pre_defined_c_library_tokens | set(vocabulary.begin_tokens) | \
                    set(vocabulary.end_tokens) | set(vocabulary.unk) | set(vocabulary.addition_tokens)
    vocabulary.special_token_ids = {vocabulary.word_to_id(t) for t in special_token}
    return vocabulary


def test_config1(is_debug):
    from model.one_pointer_copy_self_attention_seq2seq_model_gammar_mask_refactor import \
        create_parse_rnn_input_batch_data_fn, create_parse_target_batch_data, create_combine_loss_fn, create_output_ids, \
        load_dataset, RNNPointerNetworkModelWithSLKMask, slk_expand_output_and_target

    vocabulary = load_vocabulary()
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    parse_rnn_input_batch_data = create_parse_rnn_input_batch_data_fn(vocab=vocabulary)
    parse_target_batch_data = create_parse_target_batch_data()
    create_output_id_fn = create_output_ids
    loss_fn = create_combine_loss_fn(average_value=True)
    datasets = load_dataset(is_debug, vocabulary, mask_transformer=transformer)
    # datasets = load_sample_save_dataset(is_debug, vocabulary, mask_transformer=transformer)

    epoches = 80
    train_len = len(datasets[0]) if datasets[0] is not None else 100
    batch_size = 10
    clip_norm = 10

    return {
        'name': 'test',
        'save_name': 'test.pkl',
        'load_model_name': 'RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossSLKMask.pkl',

        'model_fn': RNNPointerNetworkModelWithSLKMask,
        'model_dict': {'vocabulary_size': vocabulary.vocabulary_size, 'hidden_size': 400,
                       'num_layers': 3, 'start_label': vocabulary.word_to_id(vocabulary.begin_tokens[0]),
                       'end_label': vocabulary.word_to_id(vocabulary.end_tokens[0]), 'dropout_p': 0.2,
                       'MAX_LENGTH': 500, 'atte_position_type': 'content', 'mask_transformer': transformer},

        'do_sample_evaluate': True,

        'vocabulary': vocabulary,
        # 'transformer': transformer,
        'parse_input_batch_data_fn': parse_rnn_input_batch_data,
        'parse_target_batch_data_fn': parse_target_batch_data,
        'expand_output_and_target_fn': slk_expand_output_and_target,
        'create_output_ids_fn': create_output_id_fn,
        'train_loss': loss_fn,
        'evaluate_object_list': [SLKOutputAccuracyAndCorrect(ignore_token=-1)],

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 1,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': None,


        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'do_sample_and_save': False,
        'add_data_record_fn': create_save_sample_data(vocabulary),
        'db_path': SLK_SAMPLE_DBPATH,
        'table_basename': SLK_SAMPLE_COMMON_C_ERROR_RECORDS_BASENAME,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': 0.25,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len//batch_size, 'max_grad_norm': clip_norm},
        'data': datasets

    }


def pointer_network_with_ggnn_encoder(is_debug):
    from model.one_pointer_copy_self_attention_seq2seq_model_gammar_mask_refactor import \
        create_parse_rnn_input_batch_data_fn, create_parse_target_batch_data, create_combine_loss_fn, create_output_ids, \
        load_dataset, RNNPointerNetworkModelWithSLKMask, slk_expand_output_and_target

    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                        end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                        addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    # vocabulary = load_vocabulary()
    tokenize_fn = tokenize_by_clex_fn()
    # transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)
    transformer = None

    parse_rnn_input_batch_data = create_parse_rnn_input_batch_data_fn(vocab=vocabulary, use_ast=use_ast)
    parse_target_batch_data = create_parse_target_batch_data()
    create_output_id_fn = create_output_ids
    loss_fn = create_combine_loss_fn(average_value=True)
    datasets = load_dataset(is_debug, vocabulary, mask_transformer=transformer, data_type='deepfix', use_ast=use_ast)
    # datasets = load_sample_save_dataset(is_debug, vocabulary, mask_transformer=transformer)

    epoches = 80
    train_len = len(datasets[0]) if datasets[0] is not None else 100
    batch_size = 10
    clip_norm = 10
    max_length = 500

    return {
        'name': 'pointer_network_with_ggnn_encoder_config1_position_embedding',
        'save_name': 'pointer_network_with_ggnn_encoder_config1_position_embedding.pkl',
        'load_model_name': 'pointer_network_with_ggnn_encoder_config1_position_embedding.pkl',

        'model_fn': RNNPointerNetworkModelWithSLKMask,
        'model_dict':
            {'vocabulary_size': vocabulary.vocabulary_size, 'hidden_size': 400,
             'num_layers': 3, 'start_label': vocabulary.word_to_id(vocabulary.begin_tokens[0]),
             'end_label': vocabulary.word_to_id(vocabulary.end_tokens[0]),
             'dropout_p': 0.2,
             'MAX_LENGTH': 500,
             'atte_position_type': 'content',
             'mask_transformer': transformer,
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'no_position_embedding': False,
             'position_embedding_length': 1000,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False},
             'mask_type': 'static',
             'vocabulary': vocabulary,
             },


        'do_sample_evaluate': True,

        'vocabulary': vocabulary,
        # 'transformer': transformer,
        'parse_input_batch_data_fn': parse_rnn_input_batch_data,
        'parse_target_batch_data_fn': parse_target_batch_data,
        'expand_output_and_target_fn': slk_expand_output_and_target,
        'create_output_ids_fn': create_output_id_fn,
        'train_loss': loss_fn,
        'evaluate_object_list': [SLKOutputAccuracyAndCorrect(ignore_token=-1, do_extract_evaluate=True)],

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 1,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': None,


        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'do_sample_and_save': False,
        'add_data_record_fn': create_save_sample_data(vocabulary),
        'db_path': SLK_SAMPLE_DBPATH,
        'table_basename': SLK_SAMPLE_COMMON_C_ERROR_RECORDS_BASENAME,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': 1,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': clip_norm,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len//batch_size, 'max_grad_norm': clip_norm},
        'data': datasets

    }


def encoder_copy_config1(is_debug):
    import pandas as pd
    vocabulary = load_vocabulary()
    vocabulary.add_token("<inner_begin>")
    vocabulary.add_token("<inner_end>")
    vocabulary.add_token("<PAD>")
    inner_begin_id = vocabulary.word_to_id("<inner_begin>")
    inner_end_id = vocabulary.word_to_id("<inner_end>")
    pad_id = vocabulary.word_to_id("<PAD>")
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 8
    epoches = 40
    ignore_id = -1
    max_length = 500

    if is_debug:
        from experiment.experiment_util import load_common_error_data_sample_with_encoder_copy_100
        from model.seq_to_seq_model_with_encoder_is_copy import CCodeDataset
        datasets = []
        for t in load_common_error_data_sample_with_encoder_copy_100(inner_begin_id, inner_end_id):
            t = pd.DataFrame(t)
            datasets.append(CCodeDataset(t, vocabulary, 'train', transformer,
                                         inner_begin_id, inner_end_id, begin_id, end_id, ignore_id=ignore_id,
                                         MAX_LENGTH=max_length))
        datasets.append(None)
    else:
        from experiment.experiment_util import load_common_error_data_with_encoder_copy
        from model.seq_to_seq_model_with_encoder_is_copy import CCodeDataset
        datasets = []
        for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
            t = pd.DataFrame(t)
            datasets.append(CCodeDataset(t, vocabulary, 'train', transformer,
                                         inner_begin_id, inner_end_id, begin_id, end_id, ignore_id=ignore_id,
                                         MAX_LENGTH=max_length))
        datasets.append(None)

    train_len = len(datasets[0])

    from model.seq_to_seq_model_with_encoder_is_copy import Seq2SeqEncoderCopyModel
    from model.seq_to_seq_model_with_encoder_is_copy import create_parse_target_batch_data
    from model.seq_to_seq_model_with_encoder_is_copy import create_loss_fn
    from model.seq_to_seq_model_with_encoder_is_copy import create_output_ids_fn
    return {
        'name': 'encoder_copy',
        'save_name': 'encoder_copy.pkl',
        'load_model_name': 'encoder_copy.pkl',
        'logger_file_path': 'encoder_copy.log',

        'model_fn': Seq2SeqEncoderCopyModel,
        'model_dict': {"vocabulary_size": vocabulary.vocabulary_size,
                       "embedding_size": 400,
                       "hidden_state_size": 400,
                       "start_label": begin_id,
                       "end_label": end_id,
                       "pad_label": pad_id,
                       "slk_parser": transformer,
                       "MAX_LENGTH": max_length,
                       "n_layer": 3},

        'do_sample_evaluate': False,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': lambda x, do_sample=False: [x],
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': None,
        'create_output_ids_fn': create_output_ids_fn,
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [EncoderCopyAccuracyAndCorrect(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': 0.25,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config1(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 6
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten, use_ast=False,
                                                     do_multi_step_sample=do_multi_step_sample)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                                       mask_transformer=transformer,
    #                                                                       do_flatten=do_flatten, use_ast=False,
    #                                                                       do_multi_step_sample=do_multi_step_sample)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten, use_ast=False,
    #                                                do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    return {
        'name': 'encoder_sample_dropout',
        # 'name': 'reinforcement_encoder_sample_dropout_multi_error',
        # 'save_name': 'encoder_sample_dropout_no_overfitting.pkl',
        'save_name': 'encoder_sample_dropout_overfitting.pkl',
        # 'load_model_name': 'encoder_sample_dropout_no_overfitting.pkl',
        'load_model_name': 'encoder_sample_dropout_overfitting.pkl',
        # 'logger_file_path': 'encoder_sample_dropout.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_dropout_overfitting',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict': {"start_label": begin_id,
                       "end_label": end_id,
                       "inner_start_label": inner_begin_id,
                       "inner_end_label": inner_end_id,
                       "vocabulary_size": vocabulary.vocabulary_size,
                       "embedding_size": 400,
                       "hidden_size": 400,
                       "max_sample_length": 10,
                       'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                           'max_len': max_length, 'input_size': 400,
                                           'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                           'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                           'variable_lengths': False, 'embedding': None,
                                           'update_embedding': True},
                       'graph_embedding': 'rnn',
                       'pointer_type': 'query',
                       'rnn_type': 'gru',
                       "rnn_layer_number": 3,
                       "max_length": max_length,
                       'dropout_p': 0.2,
                       'pad_label': pad_id,
                       'vocabulary': vocabulary,
                       'mask_type': 'static',
                       'beam_size': 1,
                       'p2_type': 'static',
                       'p2_step_length': 5,
                       },

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'do_beam_search': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': 1,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config2(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                                       mask_transformer=transformer,
    #                                                                       do_flatten=do_flatten, use_ast=use_ast,
    #                                                                       do_multi_step_sample=do_multi_step_sample)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        'name': 'graph_encoder_sample_config2',
        # 'name': 'graph_encoder_sample_config2_addition_data_retrain',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        'save_name': 'graph_encoder_sample_config2.pkl',
        # 'save_name': 'graph_encoder_sample_config2_addition_data_retrain.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        'load_model_name': 'graph_encoder_sample_config2.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2_addition_data_retrain.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'graph_encoder_sample_config2_24',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 10,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static'
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config3(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", True)
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                                       mask_transformer=transformer,
    #                                                                       do_flatten=do_flatten, use_ast=use_ast,
    #                                                                       do_multi_step_sample=do_multi_step_sample)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link_24',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 10,
             'graph_parameter': {"graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 },
             'graph_embedding': 'ggnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static'
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config4(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config4',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config4.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config4.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config4_17',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=True, p2_type='step'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config6(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = False
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config6_only_gru_with_token_action',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config6_only_gru_with_token_action.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config6_only_gru_with_token_action.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config6_only_gru_with_token_action_19',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
             'graph_embedding': 'rnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config7(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", True)
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config7_only_ggnn_with_token_action_with_sequence_link_25',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 },
             'graph_embedding': 'ggnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config5(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False, only_sample=True)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config5',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config5.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config5.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config5_11',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 10,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'static',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='static', only_sample=True),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=True, p2_type='static'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='static'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='static', only_sample=True),
        'train_loss': create_loss_fn(ignore_id, only_sample=True),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config8(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False, only_sample=only_sample)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config8',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config8.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config8.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config8_15',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config9(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = False
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False, only_sample=only_sample)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config9_only_gru_with_token_action_only_sample',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config9_only_gru_with_token_action_only_sample.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config9_only_gru_with_token_action_only_sample.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config9_only_gru_with_token_action_only_sample_13',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
             'graph_embedding': 'rnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config10(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", True)
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = False
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer,
                                                     do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample,
                                                     merge_action=False, only_sample=only_sample)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                           mask_transformer=transformer, do_flatten=do_flatten,
    #                                                           use_ast=use_ast,
    #                                                           do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config10_only_ggnn_with_token_action_with_sequence_link_only_sample',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config10_only_ggnn_with_token_action_with_sequence_link_only_sample.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config10_only_ggnn_with_token_action_with_sequence_link_only_sample.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'graph_encoder_sample_config3_only_ggnn_with_sequence_link',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 },
             'graph_embedding': 'ggnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step'),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step'),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config11(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = False
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer,
                                                     do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample,
                                                     merge_action=False)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                           mask_transformer=transformer, do_flatten=do_flatten,
    #                                                           use_ast=use_ast,
    #                                                           do_multi_step_sample=do_multi_step_sample,
    #                                                           customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config11',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config11.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config11.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config11_25',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 24,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config12(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = False
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config12_only_gru_with_token_action',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config12_only_gru_with_token_action.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config12_only_gru_with_token_action.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config12_only_gru_with_token_action_27',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
             'graph_embedding': 'rnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 14,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config13(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", True)
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 20
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config13_only_ggnn_with_token_action_with_sequence_link_21',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 },
             'graph_embedding': 'ggnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 2,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config14(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True
    sample_count = 1008

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False, only_sample=only_sample)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config14',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config14.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config14.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config14_20',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 12,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config15(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    sequence_output = True
    sample_count = 1008

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast, sequence_output=sequence_output)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                                       mask_transformer=transformer,
    #                                                                       do_flatten=do_flatten, use_ast=use_ast,
    #                                                                       do_multi_step_sample=do_multi_step_sample,
    #                                                                       sequence_output=sequence_output)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              sample_count=sample_count)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        'name': 'encoder_sample_config15_sequence_output',
        # 'name': 'graph_encoder_sample_config2_addition_data_retrain',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        'save_name': 'encoder_sample_config15_sequence_output.pkl',
        # 'save_name': 'graph_encoder_sample_config2_addition_data_retrain.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        'load_model_name': 'encoder_sample_config15_sequence_output.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2_addition_data_retrain.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config15_sequence_output_63',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 500,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'sequence'
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id),
        'train_loss': create_loss_fn(ignore_id, sequence_output=sequence_output),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config16(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = False
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True
    sample_count = 1008

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False, only_sample=only_sample)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config16_only_gru_with_token_action_only_sample',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config16_only_gru_with_token_action_only_sample.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config16_only_gru_with_token_action_only_sample.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'encoder_sample_config16_only_gru_with_token_action_only_sample_12',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
             'graph_embedding': 'rnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config17(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", True)
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = False
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer,
                                                     do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample,
                                                     merge_action=False, only_sample=only_sample)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                           mask_transformer=transformer, do_flatten=do_flatten,
    #                                                           use_ast=use_ast,
    #                                                           do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config17_only_ggnn_with_token_action_with_sequence_link_only_sample',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config17_only_ggnn_with_token_action_with_sequence_link_only_sample.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config17_only_ggnn_with_token_action_with_sequence_link_only_sample.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'graph_encoder_sample_config17_only_ggnn_with_sequence_link',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 },
             'graph_embedding': 'ggnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_data_generate1(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 10
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = False
    do_multi_step_sample = False
    generate_step = 5
    use_ast = False

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    from experiment.experiment_dataset import load_deepfix_ac_code_for_generate_dataset
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample)
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=True, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample)
    ac_dataset = load_deepfix_ac_code_for_generate_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=True)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten, use_ast=use_ast,
    #                                                do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from common.reinforcement_generate_util import sample_generate_action_fn
    from common.reinforcement_generate_util import calculate_encoder_sample_length_fn
    from common.reinforcement_generate_util import mask_sample_probs_with_length
    from common.reinforcement_generate_util import generate_error_code_from_ac_code_and_action_fn
    from common.reinforcement_generate_util import create_reward_by_compile
    from common.util import compile_code_ids_list
    from common.reinforcement_generate_util import create_or_sample
    from common.reinforcement_generate_util import create_output_from_actions_fn
    from common.reinforcement_generate_util import create_random_sample
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from common.reinforcement_generate_util import all_output_and_target_evaluate_fn
    return {
        # 'name': 'reinforcement_encoder_sample_dropout',
        'name': 'reinforcement_encoder_sample_dropout',
        's_saved_name': 'rl_solver_encoder_sample_dropout_multi_step.pkl',
        's_load_model_name': 'encoder_sample_dropout_no_overfitting.pkl',
        'g_saved_name': 'rl_generator_encoder_sample_dropout_multi_step.pkl',
        'g_load_model_name': 'rl_generator_encoder_sample_dropout.pkl',
        'load_previous_g_model': False,
        # 'logger_file_path': 'encoder_sample_dropout.log',

        'save_data_fn': lambda x: x,
        # 'save_data_fn': None,
        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                        use_ast=use_ast, do_multi_step_sample=False),

        'g_model_fn': EncoderSampleModel,
        'g_model_dict': {"start_label": begin_id,
                       "end_label": end_id,
                       "inner_start_label": inner_begin_id,
                       "inner_end_label": inner_end_id,
                       "vocabulary_size": vocabulary.vocabulary_size,
                       "embedding_size": 400,
                       "hidden_size": 400,
                       "max_sample_length": 3,
                       'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                           'max_len': max_length, 'input_size': 400,
                                           'input_dropout_p': 0.0, 'dropout_p': 0.0,
                                           'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                           'variable_lengths': False, 'embedding': None,
                                           'update_embedding': True},
                       'graph_embedding': 'rnn',
                       'pointer_type': 'query',
                       'rnn_type': 'gru',
                       "rnn_layer_number": 3,
                       "max_length": max_length,
                       'dropout_p': 0,
                       'pad_label': pad_id,
                       'vocabulary': vocabulary,
                       'mask_type': 'static',
                       'beam_size': 1,
                       'p2_type': 'static',
                       'p2_step_length': -1,
                       },

        'agent_dict': {
            'parse_input_batch_data_fn': create_parse_input_batch_data_fn(),
            'sample_generate_action_fn': sample_generate_action_fn(create_output_from_actions_fn(create_or_sample),
                                                                   calculate_encoder_sample_length_fn(inner_end_id),
                                                                   mask_sample_probs_with_length,
                                                                   init_explore_p=0.1,
                                                                   min_explore_p=0.001,
                                                                   decay_step=10000,
                                                                   decay=0.2,
                                                                   p2_step_length=3),
            'do_sample': True,
            'do_beam_search': False,
            'reward_discount_gamma': 0.99,
            'do_normalize': False,
        },
        'random_agent_dict': {
            'parse_input_batch_data_fn': create_parse_input_batch_data_fn(),
            'sample_generate_action_fn': sample_generate_action_fn(create_output_from_actions_fn(create_random_sample),
                                                                   calculate_encoder_sample_length_fn(inner_end_id),
                                                                   mask_sample_probs_with_length,
                                                                   init_explore_p=0.1,
                                                                   min_explore_p=0.001,
                                                                   decay_step=10000,
                                                                   decay=0.2,
                                                                   p2_step_length=3),
            'do_sample': True,
            'do_beam_search': False,
            'reward_discount_gamma': 0.99,
            'do_normalize': False,
        },
        'do_random_generate': True,
        'generate_step': generate_step,

        's_model_fn': EncoderSampleModel,
        's_model_dict': {"start_label": begin_id,
                       "end_label": end_id,
                       "inner_start_label": inner_begin_id,
                       "inner_end_label": inner_end_id,
                       "vocabulary_size": vocabulary.vocabulary_size,
                       "embedding_size": 400,
                       "hidden_size": 400,
                       "max_sample_length": 10,
                       'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                           'max_len': max_length, 'input_size': 400,
                                           'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                           'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                           'variable_lengths': False, 'embedding': None,
                                           'update_embedding': True},
                       'graph_embedding': 'rnn',
                       'pointer_type': 'query',
                       'rnn_type': 'gru',
                       "rnn_layer_number": 3,
                       "max_length": max_length,
                       'dropout_p': 0.2,
                       'pad_label': pad_id,
                       'vocabulary': vocabulary,
                       'mask_type': 'static',
                       'beam_size': 1,
                       'p2_type': 'static',
                       'p2_step_length': 5,
                       },

        'environment_dict': {'batch_size': batch_size,
                             'preprocess_next_input_for_solver_fn': generate_error_code_from_ac_code_and_action_fn(
                                 inner_end_id, begin_id, end_id, vocabulary=vocabulary, use_ast=use_ast),
                             'parse_input_batch_data_for_solver_fn': create_parse_input_batch_data_fn(),
                             'solver_create_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id),
                             'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_token=ignore_id),
                             'create_records_all_output_fn': create_records_all_output,
                             'evaluate_output_result_fn': all_output_and_target_evaluate_fn(ignore_token=ignore_id),
                             'vocabulary': vocabulary,
                             'compile_code_ids_fn': compile_code_ids_list,
                             'extract_includes_fn': lambda x: x['includes'],
                             'create_reward_by_compile_fn': create_reward_by_compile,
                             'data_radio': 1.0,
                             'inner_begin_label': inner_begin_id,
                             'inner_end_label': inner_end_id,
                             'use_ast': use_ast
                             },

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'do_beam_search': False,
        'g_max_step_times': 3,
        's_max_step_times': 3,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': 1,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        's_optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len//batch_size, 'max_grad_norm': 10},
        'g_optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len / generate_step // batch_size, 'max_grad_norm': 10},
        'data': datasets,
        'ac_data': ac_dataset,
    }


def encoder_sample_data_generate2(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 10
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = False
    do_multi_step_sample = False
    generate_step = 2
    if is_debug:
        fast_ac_data_len = 50
        fast_train_len = 50
    else:
        fast_ac_data_len = 3000
        fast_train_len = 3000

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    from experiment.experiment_dataset import load_deepfix_ac_code_for_generate_dataset
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample)
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=True, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample)
    ac_dataset = load_deepfix_ac_code_for_generate_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=True)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten, use_ast=use_ast,
    #                                                do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from common.reinforcement_generate_util import sample_generate_action_fn
    from common.reinforcement_generate_util import calculate_encoder_sample_length_fn
    from common.reinforcement_generate_util import mask_sample_probs_with_length
    from common.reinforcement_generate_util import generate_error_code_from_ac_code_and_action_fn
    from common.reinforcement_generate_util import create_reward_by_compile
    from common.util import compile_code_ids_list
    from common.reinforcement_generate_util import create_or_sample
    from common.reinforcement_generate_util import create_output_from_actions_fn
    from common.reinforcement_generate_util import create_random_sample
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from common.reinforcement_generate_util import all_output_and_target_evaluate_fn
    return {
        # 'name': 'reinforcement_encoder_sample_dropout',
        'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        's_saved_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 's_load_model_name': 'encoder_sample_dropout_no_overfitting.pkl',
        # 's_load_model_name': 'graph_encoder_sample_config2.pkl',
        's_load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        'g_saved_name': 'rl_generator_graph_encoder_sample_config2_fast_iterate.pkl',
        'g_load_model_name': 'rl_generator_graph_encoder_sample_config2_fast_iterate.pkl',
        'load_previous_g_model': True,
        # 'logger_file_path': 'encoder_sample_dropout.log',

        'save_data_fn': lambda x: x,
        # 'save_data_fn': None,
        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                        use_ast=use_ast, do_multi_step_sample=False),
        'only_cant_fix': True,
        'max_generate_distance': 15,

        'g_model_fn': EncoderSampleModel,
        'g_model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 3,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static'
             },

        'agent_dict': {
            'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast),
            'sample_generate_action_fn': sample_generate_action_fn(create_output_from_actions_fn(create_or_sample),
                                                                   calculate_encoder_sample_length_fn(inner_end_id),
                                                                   mask_sample_probs_with_length,
                                                                   init_explore_p=0.1,
                                                                   min_explore_p=0.001,
                                                                   decay_step=10000,
                                                                   decay=0.2,
                                                                   p2_step_length=3),
            'do_sample': True,
            'do_beam_search': False,
            'reward_discount_gamma': 0.99,
            'do_normalize': False,
        },
        'random_agent_dict': {
            'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast),
            'sample_generate_action_fn': sample_generate_action_fn(create_output_from_actions_fn(create_random_sample),
                                                                   calculate_encoder_sample_length_fn(inner_end_id),
                                                                   mask_sample_probs_with_length,
                                                                   init_explore_p=0.1,
                                                                   min_explore_p=0.001,
                                                                   decay_step=10000,
                                                                   decay=0.2,
                                                                   p2_step_length=3),
            'do_sample': True,
            'do_beam_search': False,
            'reward_discount_gamma': 0.99,
            'do_normalize': False,
        },
        'do_random_generate': False,
        'generate_step': generate_step,

        's_model_fn': EncoderSampleModel,
        's_model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 10,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static'
             },

        'environment_dict': {'batch_size': batch_size,
                             'preprocess_next_input_for_solver_fn': generate_error_code_from_ac_code_and_action_fn(
                                 inner_end_id, begin_id, end_id, vocabulary=vocabulary, use_ast=use_ast),
                             'parse_input_batch_data_for_solver_fn': create_parse_input_batch_data_fn(use_ast=use_ast),
                             'solver_create_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id,
                                                                                                        end_id, inner_end_id,
                                                                                                        vocabulary=vocabulary,
                                                                                                        use_ast=use_ast),
                             'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_token=ignore_id),
                             'create_records_all_output_fn': create_records_all_output,
                             'evaluate_output_result_fn': all_output_and_target_evaluate_fn(ignore_token=ignore_id),
                             'vocabulary': vocabulary,
                             'compile_code_ids_fn': compile_code_ids_list,
                             'extract_includes_fn': lambda x: x['includes'],
                             'create_reward_by_compile_fn': create_reward_by_compile,
                             'data_radio': 1.0,
                             'inner_begin_label': inner_begin_id,
                             'inner_end_label': inner_end_id,
                             'use_ast': use_ast
                             },

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'do_beam_search': False,
        'g_max_step_times': 3,
        's_max_step_times': 3,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'do_tree_generate': False,

        'do_fast_generate': True,
        'fast_train_len': fast_train_len,
        'fast_ac_data_len': fast_ac_data_len,

        'epcohes': epoches,
        'start_epoch': 80,
        'epoch_ratio': 1,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        's_optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len//batch_size, 'max_grad_norm': 10},
        'g_optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoches * train_len / generate_step // batch_size, 'max_grad_norm': 10},
        'data': datasets,
        'ac_data': ac_dataset,
    }


def sensibility_rnn_config1(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                        end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                        addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = False
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    do_multi_step_sample = False

    ignore_id = -1
    max_length = 500
    epoches = 120
    batch_size = 32

    from experiment.experiment_dataset import load_deepfix_ac_code_for_generate_dataset
    from experiment.experiment_dataset import load_deepfix_ac_code_for_sensibility_rnn
    from model.sensibility_baseline.rnn_pytorch import rnn_parse_input_batch_data_fn
    from model.sensibility_baseline.rnn_pytorch import rnn_parse_target_batch_data_fn
    from model.sensibility_baseline.rnn_pytorch import SensibilityBiRnnModel
    from model.sensibility_baseline.rnn_pytorch import create_loss_function
    datasets = load_deepfix_ac_code_for_sensibility_rnn(is_debug=is_debug, vocabulary=vocabulary,
                                                           mask_transformer=None, do_flatten=True,
                                                           use_ast=use_ast,
                                                           do_multi_step_sample=False)
    train_len = len(datasets[0]) if datasets[0] is not None else 100

    from model.sensibility_baseline.rnn_pytorch import create_output_fn
    return {
        'name': 'sensibility_rnn_config1',
        'save_name': 'sensibility_rnn_config1.pkl',
        'load_model_name': 'sensibility_rnn_config1.pkl',

        'save_data_fn': lambda x: x,

        'model_fn': SensibilityBiRnnModel,
        'model_dict':
            {"vocabulary_size": vocabulary.vocabulary_size,
             "embedding_dim": 400,
             "hidden_size": 400,
             'encoder_params': {'vocab_size': vocabulary.vocabulary_size,
                                   'max_len': max_length, 'input_size': 400,
                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                   'n_layers': 3, 'bidirectional': False, 'rnn_cell': 'gru',
                                   'variable_lengths': False, 'embedding': None,
                                   'update_embedding': True,
                                 },
             },

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'create_multi_step_next_input_batch_fn': None,
        'max_step_times': 10,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': None,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': rnn_parse_input_batch_data_fn(),
        'parse_target_batch_data_fn': rnn_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': None,
        'create_output_ids_fn': create_output_fn,
        'train_loss': create_loss_function(ignore_id),
        'evaluate_object_list': [SensibilityRNNEvaluator(ignore_token=ignore_id)],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': 1,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                             't_total': epoches * train_len // batch_size, 'max_grad_norm': 10},
        'data': datasets,
    }


def sensibility_rnn_config2(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                        end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                        addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])

    use_ast = False

    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 1.0
    addition_step = 3
    only_sample = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test

    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    train_len = len(datasets[0]) if datasets[0] is not None else 100

    from model.sensibility_baseline.rnn_pytorch import SensibilityBiRnnModel
    rnn_model = SensibilityBiRnnModel(**{"vocabulary_size": vocabulary.vocabulary_size,
                                         "embedding_dim": 400,
                                         "hidden_size": 400,
                                         'encoder_params': {'vocab_size': vocabulary.vocabulary_size,
                                                            'max_len': max_length, 'input_size': 400,
                                                            'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                            'n_layers': 3, 'bidirectional': False, 'rnn_cell': 'gru',
                                                            'variable_lengths': False, 'embedding': None,
                                                            'update_embedding': True,
                                                            },
                                         })
    from common import torch_util
    import os
    rnn_path = os.path.join(save_model_root, "rnn_baseline", "sensibility_rnn_config1.pkl81")
    from torch import nn
    rnn_model = nn.DataParallel(rnn_model.cuda(), device_ids=[0])
    # torch_util.load_model(rnn_model, rnn_path, map_location={'cuda:1': 'cuda:0'})
    torch_util.load_model(rnn_model, rnn_path)
    rnn_model = rnn_model.module.cpu()

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.sensibility_baseline.baseline_model import FixModel
    from model.sensibility_baseline.baseline_model import parse_input_batch_data_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'sensibility_rnn_config2',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'sensibility_rnn_config2.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'sensibility_rnn_config2.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'sensibility_rnn_config2_81',
        'change_output_records_to_batch_fn': change_output_records_to_batch_fn(one_step=True),
        'create_save_database_records_fn': sensibility_create_save_database_records,

        'model_fn': FixModel,
        'model_dict':
            {
                "rnn_model": rnn_model,
                "vocabulary": vocabulary,
            },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 1,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step', only_sample=only_sample,
                                                                                       one_step=True),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': parse_input_batch_data_fn,
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step', only_sample=only_sample),
        'train_loss': create_loss_fn(ignore_id, only_sample=only_sample),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config18(is_debug):
    vocabulary = create_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    error_stop_type = 'normal'
    epoch_ratio = 0.4
    addition_step = 3
    sample_count = 1008

    from experiment.experiment_dataset import load_common_error_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_common_error_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config18',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config18.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config18.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_CODEFORCES_TRAIN_DBPATH,
        'table_basename': 'encoder_sample_config18_24',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def encoder_sample_config19(is_debug):
    vocabulary = create_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = False
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    epoch_ratio = 0.4
    addition_step = 3

    from experiment.experiment_dataset import load_common_error_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_common_error_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        # 'name': 'graph_encoder_sample_config2',
        'name': 'encoder_sample_config19',
        # 'name': 'reinforcement_graph_encoder_sample_config2_fast_iterate',
        # 'save_name': 'graph_encoder_sample_config2.pkl',
        'save_name': 'encoder_sample_config19.pkl',
        # 'save_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'load_model_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config19.pkl',
        # 'load_model_name': 'rl_solver_graph_encoder_sample_config2_fast_iterate.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_CODEFORCES_TRAIN_DBPATH,
        'table_basename': 'encoder_sample_config19_24',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 3, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
             'graph_embedding': 'rnn',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


# ---------------------------------------------- Pretrain Model Experiment Config ----------------------------------- #

def pretrain_encoder_sample_no_pretrain_config1(is_debug):
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    return {
        'name': 'pretrain_encoder_sample_no_pretrain_config1',
        'save_name': 'pretrain_encoder_sample_no_pretrain_config1.pkl',
        'load_model_name': 'pretrain_encoder_sample_no_pretrain_config1.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_no_pretrain_config1_30',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def pretrain_encoder_sample_config2(is_debug):
    # pretrain config name: pretrain_masked_model2.pkl13
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008
    check_error_task = True
    train_type = 'only_disc'

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.pretrain_model.generate_detect_model import PretrainMaskedCodeModel
    return {
        'name': 'pretrain_encoder_sample_config2',
        'save_name': 'pretrain_encoder_sample_config2_pretrain13.pkl',
        'load_model_name': 'pretrain_encoder_sample_config2_pretrain13.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'load_pretrain_model': False,
        'pretrain_model_fn': PretrainMaskedCodeModel,
        'pretrain_model_params': {
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
        },
        'pretrain_model_path': 'pretrain_masked_model2.pkl13',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_config2_pretrain13_30',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def pretrain_encoder_sample_config1(is_debug):
    # pretrain config name: pretrain_masked_model1.pkl41
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008
    check_error_task = True
    train_type = 'both'

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.pretrain_model.generate_detect_model import PretrainMaskedCodeModel
    return {
        'name': 'pretrain_encoder_sample_config1',
        'save_name': 'pretrain_encoder_sample_config1_pretrain41.pkl',
        'load_model_name': 'pretrain_encoder_sample_config1_pretrain41.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'load_pretrain_model': False,
        'pretrain_model_fn': PretrainMaskedCodeModel,
        'pretrain_model_params': {
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
        },
        'pretrain_model_path': 'pretrain_masked_model1.pkl41',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_config1_pretrain41_32',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def pretrain_encoder_sample_config4(is_debug):
    # pretrain config name: pretrain_masked_model4.pkl38
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008
    check_error_task = True
    train_type = 'both'

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.pretrain_model.generate_detect_model import PretrainMaskedCodeModel
    return {
        'name': 'pretrain_encoder_sample_config4',
        'save_name': 'pretrain_encoder_sample_config4_gene20_pretrain38.pkl',
        'load_model_name': 'pretrain_encoder_sample_config4_gene20_pretrain38.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'load_pretrain_model': False,
        'pretrain_model_fn': PretrainMaskedCodeModel,
        'pretrain_model_params': {
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
        },
        'pretrain_model_path': 'pretrain_masked_model4_gene20.pkl38',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_config4_gene20_pretrain38_27',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def pretrain_encoder_sample_config5(is_debug):
    # pretrain config name: pretrain_masked_model5.pkl35
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = True
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008
    check_error_task = False
    train_type = 'bert'

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer,
    #                                                  do_flatten=do_flatten, use_ast=use_ast,
    #                                                  do_multi_step_sample=do_multi_step_sample,
    #                                                  merge_action=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                              mask_transformer=transformer, do_flatten=do_flatten,
                                                              use_ast=use_ast,
                                                              do_multi_step_sample=do_multi_step_sample,
                                                              customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.pretrain_model.generate_detect_model import PretrainMaskedCodeModel
    return {
        'name': 'pretrain_encoder_sample_config5',
        'save_name': 'pretrain_encoder_sample_config5_pretrain35.pkl',
        'load_model_name': 'pretrain_encoder_sample_config5_pretrain35.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'load_pretrain_model': False,
        'pretrain_model_fn': PretrainMaskedCodeModel,
        'pretrain_model_params': {
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
        },
        'pretrain_model_path': 'pretrain_masked_model5.pkl35',

        'do_save_records_to_database': True,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_config5_pretrain35_31',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def pretrain_encoder_sample_config6(is_debug):
    # pretrain config name: pretrain_masked_model6.pkl
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = False
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008
    check_error_task = False
    train_type = 'bert'

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer,
                                                     do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample,
                                                     merge_action=False)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                           mask_transformer=transformer, do_flatten=do_flatten,
    #                                                           use_ast=use_ast,
    #                                                           do_multi_step_sample=do_multi_step_sample,
    #                                                           customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.pretrain_model.generate_detect_model import PretrainMaskedCodeModel
    return {
        'name': 'pretrain_encoder_sample_config6',
        'save_name': 'pretrain_encoder_sample_config6_pretrain33.pkl',
        'load_model_name': 'pretrain_encoder_sample_config6_pretrain33.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'load_pretrain_model': True,
        'pretrain_model_fn': PretrainMaskedCodeModel,
        'pretrain_model_params': {
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
        },
        'pretrain_model_path': 'pretrain_masked_model6.pkl33',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_config6_pretrain33_',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }


def pretrain_encoder_sample_config7(is_debug):
    # pretrain config name: pretrain_masked_model7.pkl
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>', '<MASK>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = True
    if use_ast:
        from experiment.experiment_dataset import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 40
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    do_multi_step_sample = False
    error_stop_type = 'normal'
    epoch_ratio = 1.0
    addition_step = 3
    sample_count = 1008
    check_error_task = False
    train_type = 'both'

#     code = r'''
#     #include<stdio.h>
#
# int main (){
#
#     int a,b;
#     scanf("%d %d", &a, &b);
#     if (a == 0)
#         b = b + 1;
#     }
#     int c=a+b;
#     printf("%d\n", c);
# }
#     '''
    code = r'''
    #include<stdio.h>
        int main ( ) {
  int n , r [ 100 ] , t , i ;
  scanf ( "%d" , & n ) ;
  r [ 0 ] = 1 ;
  for ( i = 1 ; i < n ; i ++ ) {
     scanf ( "%d" , & r [ i ] ) ;
  }
  for ( i = 0 ; i < n ; i ++ ) {
    printf ( "%d" , r[ i ] ) ;
  }
}
return 0 ;
}
    '''
#     code = '''
#     int main()
#         int a = 0 ;
#         a = a + 1 ;
#     }
#     '''
#     df = pd.DataFrame({'code_id': ['test'], 'code': [code], 'errorcount': [1]})

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten,
    #                                                  use_ast=use_ast)
    # from experiment.experiment_dataset import load_deepfix_flatten_combine_node_sample_iterative_dataset
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer,
                                                     do_flatten=do_flatten, use_ast=use_ast,
                                                     do_multi_step_sample=do_multi_step_sample,
                                                     merge_action=False)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                           mask_transformer=transformer, do_flatten=do_flatten,
    #                                                           use_ast=use_ast,
    #                                                           do_multi_step_sample=do_multi_step_sample,
    #                                                           customer_df=None, sample_count=None)

    # if is_debug:
    #     from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, load_fake_deepfix_dataset_iterate_error_data_sample_100
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length, do_flatten=do_flatten))
    #     datasets.append(None)
    # else:
    #     from experiment.experiment_util import load_common_error_data_with_encoder_copy
    #     from experiment.experiment_dataset import IterateErrorDataSet
    #     datasets = []
    #     for t in load_common_error_data_with_encoder_copy(inner_begin_id, inner_end_id):
    #         t = pd.DataFrame(t)
    #         datasets.append(IterateErrorDataSet(t, vocabulary, 'train', transformer, MAX_LENGTH=max_length))
    #     datasets.append(None)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.encoder_sample_model import EncoderSampleModel
    from model.encoder_sample_model import create_parse_target_batch_data
    from model.encoder_sample_model import create_loss_fn
    from model.encoder_sample_model import create_output_ids_fn
    from model.encoder_sample_model import expand_output_and_target_fn
    from model.encoder_sample_model import create_multi_step_next_input_batch_fn
    from model.encoder_sample_model import multi_step_print_output_records_fn
    from experiment.experiment_dataset import load_addition_generate_iterate_solver_train_dataset_fn
    from model.pretrain_model.generate_detect_model import PretrainMaskedCodeModel
    return {
        'name': 'pretrain_encoder_sample_config7',
        'save_name': 'pretrain_encoder_sample_config7_pretrain45.pkl',
        'load_model_name': 'pretrain_encoder_sample_config6_pretrain45.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'load_pretrain_model': True,
        'pretrain_model_fn': PretrainMaskedCodeModel,
        'pretrain_model_params': {
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
        },
        'pretrain_model_path': 'pretrain_masked_model7.pkl45',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_DBPATH,
        'table_basename': 'pretrain_encoder_sample_config7_pretrain45_',
        'change_output_records_to_batch_fn': change_output_records_to_batch,
        'create_save_database_records_fn': create_save_database_records,

        'model_fn': EncoderSampleModel,
        'model_dict':
            {"start_label": begin_id,
             "end_label": end_id,
             "inner_start_label": inner_begin_id,
             "inner_end_label": inner_end_id,
             "vocabulary_size": vocabulary.vocabulary_size,
             "embedding_size": 400,
             "hidden_size": 400,
             "max_sample_length": 1,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             'graph_embedding': 'mixed',
             'pointer_type': 'query',
             'rnn_type': 'gru',
             "rnn_layer_number": 3,
             "max_length": max_length,
             'dropout_p': 0.2,
             'pad_label': pad_id,
             'vocabulary': vocabulary,
             'mask_type': 'static',
             'p2_type': 'step',
             'p2_step_length': 2,
             'feedforward_output': True,
             },

        'random_embedding': False,
        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': do_multi_step_sample,
        'error_stop_type': error_stop_type,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id,
                                                                                       vocabulary=vocabulary, use_ast=use_ast,
                                                                                       p2_type='step'),
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',
        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

        'load_addition_generate_iterate_solver_train_dataset_fn':
            load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, transformer, do_flatten=True,
                                                                   use_ast=use_ast, do_multi_step_sample=False),
        'max_save_distance': 15,
        'addition_train': False,
        'addition_step': addition_step,
        'no_addition_step': 10,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(use_ast=use_ast, p2_type='step', feedforward_output=True),
        'parse_target_batch_data_fn': create_parse_target_batch_data(ignore_id, p2_type='step', feedforward_output=True),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(inner_end_id, p2_type='step'),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [ErrorPositionAndValueAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }

