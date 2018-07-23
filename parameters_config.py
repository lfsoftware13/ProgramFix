from common.constants import pre_defined_c_tokens, pre_defined_c_library_tokens, SLK_SAMPLE_DBPATH, \
    SLK_SAMPLE_COMMON_C_ERROR_RECORDS_BASENAME
from common.evaluate_util import SLKOutputAccuracyAndCorrect
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from model.one_pointer_copy_self_attention_seq2seq_model_gammar_mask_refactor import load_sample_save_dataset, \
    create_save_sample_data
from read_data.load_data_vocabulary import create_common_error_vocabulary
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
    # datasets = load_dataset(is_debug, vocabulary, mask_transformer=transformer)
    datasets = load_sample_save_dataset(is_debug, vocabulary, mask_transformer=transformer)

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
                       'end_label': vocabulary.word_to_id(vocabulary.end_tokens[0]), 'dropout_p': 0,
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

        'do_sample_and_save': True,
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



