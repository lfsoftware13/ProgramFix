import os

from common.pycparser_util import tokenize_by_clex_fn
from common.util import ensure_file_path
from config import train_ids_path, test_ids_path, valid_ids_path
from experiment.experiment_dataset import load_deepfix_sample_iterative_dataset
from read_data.load_data_vocabulary import create_deepfix_common_error_vocabulary
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK


def read_deepfix_dataset():
    tokenize_fn = tokenize_by_clex_fn()
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                        end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                        addition_tokens=['<PAD>'])
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocabulary)
    train_dataset, valid_dataset, test_dataset, _ = load_deepfix_sample_iterative_dataset(is_debug=False, vocabulary=vocabulary,
                                                     mask_transformer=transformer,
                                                     do_flatten=True, use_ast=True,
                                                     do_multi_step_sample=False,
                                                     merge_action=False)
    return train_dataset, valid_dataset, test_dataset


def extract_record_id(dataset):
    ids = [one['id'] for one in dataset._samples]
    ids = list(set(ids))
    return ids


def save_ids_to_file(ids, save_path):
    ensure_file_path(save_path)
    with open(save_path, mode='w') as f:
        id_line = ','.join(ids)
        f.write(id_line)
    print('save {} ids to {}'.format(len(ids), save_path))


def save_deepfix_data_id_main():
    train_dataset, valid_dataset, test_dataset = read_deepfix_dataset()
    train_ids = extract_record_id(train_dataset)
    valid_ids = extract_record_id(valid_dataset)
    test_ids = extract_record_id(test_dataset)
    save_ids_to_file(train_ids, train_ids_path)
    save_ids_to_file(valid_ids, valid_ids_path)
    save_ids_to_file(test_ids, test_ids_path)


if __name__ == '__main__':
    save_deepfix_data_id_main()