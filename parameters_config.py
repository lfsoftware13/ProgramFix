from common.constants import pre_defined_c_tokens, pre_defined_c_library_tokens, \
    SLK_SAMPLE_COMMON_C_ERROR_RECORDS_BASENAME
from config import SLK_SAMPLE_DBPATH
from common.evaluate_util import SLKOutputAccuracyAndCorrect, EncoderCopyAccuracyAndCorrect, \
    ErrorPositionAndValueAccuracy
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from model.encoder_sample_model import create_parse_input_batch_data_fn
from model.one_pointer_copy_self_attention_seq2seq_model_gammar_mask_refactor import load_sample_save_dataset, \
    create_save_sample_data
from read_data.load_data_vocabulary import create_common_error_vocabulary, create_deepfix_common_error_vocabulary
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK


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

    epoches=40
    train_len = len(datasets[0])
    batch_size = 2
    clip_norm = 1

    return {
        'name': 'test',
        'save_name': 'test.pkl',
        'load_model_name': 'RNNPointerAllLossWithContentEmbeddingCombineTrainWeightCopyLossSLKMask.pkl',

        'model_fn': RNNPointerNetworkModelWithSLKMask,
        'model_dict': {'vocabulary_size': vocabulary.vocabulary_size, 'hidden_size': 400,
                       'num_layers': 3, 'start_label': vocabulary.word_to_id(vocabulary.begin_tokens[0]),
                       'end_label': vocabulary.word_to_id(vocabulary.end_tokens[0]), 'dropout_p': 0.2,
                       'MAX_LENGTH': 500, 'atte_position_type': 'content', 'mask_transformer': transformer},

        'do_sample_evaluate': False,

        'vocabulary': vocabulary,
        # 'transformer': transformer,
        'parse_input_batch_data_fn': parse_rnn_input_batch_data,
        'parse_target_batch_data_fn': parse_target_batch_data,
        'expand_output_and_target_fn': slk_expand_output_and_target,
        'create_output_ids_fn': create_output_id_fn,
        'train_loss': loss_fn,
        'evaluate_object_list': [SLKOutputAccuracyAndCorrect(ignore_token=-1)],

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

    batch_size = 10
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    # datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten, use_ast=False)
    datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten, use_ast=False)

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
        'save_name': 'encoder_sample_dropout.pkl',
        'load_model_name': 'encoder_sample_dropout_no_overfitting.pkl',
        'logger_file_path': 'encoder_sample_dropout.log',

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
                       },

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': True,
        'do_beam_search': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id),
        'compile_file_path': 'tmp/main.c',
        'target_file_path': 'tmp/main.out',
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
        'epoch_ratio': 0.35,
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
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    do_flatten = True
    epoch_ratio = 0.35

    from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset, \
        load_deeffix_error_iterative_dataset_real_test
    datasets = load_deepfix_sample_iterative_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                                     mask_transformer=transformer, do_flatten=do_flatten,
                                                     use_ast=True)
    # datasets = load_deeffix_error_iterative_dataset_real_test(vocabulary=vocabulary,
    #                                                  mask_transformer=transformer, do_flatten=do_flatten)

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
    return {
        'name': 'encoder_sample_config2',
        'save_name': 'encoder_sample_config2.pkl',
        'load_model_name': 'encoder_sample_config2.pkl',
        'logger_file_path': 'encoder_sample_config2.log',

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
                                                   'update_embedding': True},
                                 "graph_type": "ggnn",
                                 "graph_itr": 3},
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


        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': create_multi_step_next_input_batch_fn(begin_id, end_id, inner_end_id),
        'compile_file_path': 'tmp/main.c',
        'extract_includes_fn': lambda x: x['includes'],
        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(inner_end_id),

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

