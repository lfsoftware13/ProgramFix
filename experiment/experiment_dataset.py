import random

from common.pycparser_util import tokenize_by_clex_fn
from common.util import CustomerDataSet, show_process_map
from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data_sample_100, \
    load_fake_deepfix_dataset_iterate_error_data, load_deepfix_error_data_for_iterate
from read_data.load_data_vocabulary import create_deepfix_common_error_vocabulary
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK
from vocabulary.word_vocabulary import Vocabulary

import pandas as pd


MAX_LENGTH = 500

class IterateErrorDataSet(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transformer_vocab_slk=None,
                 no_filter=False,
                 do_flatten=False,
                 MAX_LENGTH=500):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        self.is_flatten = do_flatten
        self.max_length = MAX_LENGTH
        self.transform = False
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df
            self._samples = [row for i, row in self.data_df.iterrows()]
            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)
            # for s in self._samples:
            #     for k, v in s.items():
            #         print("{}:shape {}".format(k, np.array(v).shape))

    def filter_df(self, df):
        df = df[df['error_token_id_list'].map(lambda x: x is not None)]
        # print('CCodeErrorDataSet df before: {}'.format(len(df)))
        df = df[df['distance'].map(lambda x: x >= 0)]
        # print('CCodeErrorDataSet df after: {}'.format(len(df)))
        # print(self.data_df['error_code_word_id'])
        # df['error_code_word_id'].map(lambda x: print(type(x)))
        # print(df['error_code_word_id'].map(lambda x: print(x))))
        def iterate_check_max_len(x):
            if not self.is_flatten:
                for i in x:
                    if len(i) > self.max_length:
                        return False
                return True
            else:
                return len(x) < self.max_length
        df = df[df['error_token_id_list'].map(iterate_check_max_len)]
        end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])

        slk_count = 0
        error_count = 0
        not_in_mask_error = 0
        def filter_slk(ac_code_ids):
            res = None
            cur_ac_code_ids = ac_code_ids[1:-1]
            nonlocal slk_count, error_count, not_in_mask_error
            slk_count += 1
            if slk_count%100 == 0:
                print('slk count: {}, error count: {}, not in mask error: {}'.format(slk_count, error_count, not_in_mask_error))
            try:
                mask_list = self.transformer.get_all_token_mask_train([cur_ac_code_ids])[0]
                res = True
                for one_id, one_mask in zip(cur_ac_code_ids+[end_id], mask_list):
                    if one_id not in one_mask:
                        # print(' '.join([self.vocabulary.id_to_word(i) for i in ac_code_ids]))
                        # print('id {} not in mask {}'.format(one_id, one_mask))
                        not_in_mask_error += 1
                        res = None
                        break
            except Exception as e:
                error_count += 1
                # print('slk error : {}'.format(str(e)))
                res = None
            return res

        # tmp comment the slk filter
        # print('before slk grammar: {}'.format(len(df)))
        # df['grammar_mask_list'] = df['ac_code_ids'].map(filter_slk)
        # df = df[df['grammar_mask_list'].map(lambda x: x is not None)]
        # print('after slk grammar: {}'.format(len(df)))

        return df

    def _get_raw_sample(self, row):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]
        # sample = dict(row)
        sample = {}
        sample['input_seq'] = row['error_token_id_list']
        sample['includes'] = row['includes']
        if not self.is_flatten:
            sample['input_length'] = [len(ids) for ids in sample['input_seq']]
        else:
            sample['input_length'] = len(sample['input_seq'])

        inner_begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[1])
        inner_end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[1])
        if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':

            # sample['sample_inputs'] = [[inner_begin_id]+one for one in row['sample_ac_id_list']]
            # sample['sample_inputs_length'] = [len(ids) for ids in sample['sample_inputs']]
            if not self.is_flatten:
                sample['is_copy_target'] = [one + [0] for one in row['is_copy_list']]
                sample['copy_target'] = [one + [-1] for one in row['copy_pos_list']]
                # sample['copy_length'] = [len(one) for one in sample['is_copy_target']]

                # sample['sample_target'] = [one + [inner_end_id] for one in row['sample_ac_id_list']]
                sample['sample_outputs_length'] = [len(ids) for ids in sample['sample_target']]
                sample['target'] = [one + [inner_end_id] for one in row['sample_ac_id_list']]

                error_start_pos_list, error_end_pos_list = list(*row['error_pos_list'])
                sample['p1_target'] = error_start_pos_list
                sample['p2_target'] = error_end_pos_list
                sample['error_pos_list'] = row['error_pos_list']

                sample['compatible_tokens'] = [row['sample_mask_list'] + [inner_end_id]
                                              for i in range(len(sample['sample_target']))]
                sample['compatible_tokens_length'] = [len(one) for one in sample['compatible_tokens']]

                sample['distance'] = row['distance']
                sample['adj'] = 0
            else:
                sample['target'] = [inner_begin_id] + row['sample_ac_id_list'] + [inner_end_id]

                sample['is_copy_target'] = row['is_copy_list'] + [0]
                sample['copy_target'] = row['copy_pos_list'] + [-1]
                sample['copy_length'] = sample['input_length']

                sample_mask = sorted(row['sample_mask_list'] + [inner_end_id])
                sample_mask_dict = {v: i for i, v in enumerate(sample_mask)}
                sample['compatible_tokens'] = [sample_mask for i in range(len(sample['is_copy_target']))]
                sample['compatible_tokens_length'] = [len(one) for one in sample['compatible_tokens']]

                sample['sample_target'] = row['sample_ac_id_list'] + [inner_end_id]
                sample['sample_target'] = [t if c == 0 else -1 for c, t in zip(sample['is_copy_target'], sample['sample_target'])]
                sample['sample_small_target'] = [sample_mask_dict[t] if c == 0 else -1 for c, t in zip(sample['is_copy_target'], sample['sample_target'])]
                sample['sample_outputs_length'] = len(sample['sample_target'])

                sample['full_output_target'] = row['target_ac_token_id_list'][1:-1]

                sample['final_output'] = row['ac_code_ids']
                sample['p1_target'] = row['error_pos_list'][0]
                sample['p2_target'] = row['error_pos_list'][1]
                sample['error_pos_list'] = row['error_pos_list']

                sample['distance'] = row['distance']
                sample['includes'] = row['includes']
                sample['adj'] = 0
        else:
            sample['copy_length'] = sample['input_length']
            sample['adj'] = 0
        return sample

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [row for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = IterateErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                              transformer_vocab_slk=self.transformer)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = IterateErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                                transformer_vocab_slk=self.transformer)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __len__(self):
        return len(self._samples)


def load_deepfix_sample_iterative_dataset(is_debug, vocabulary, mask_transformer, do_flatten=False):
    if is_debug:
        data_dict = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten)
    else:
        data_dict = load_fake_deepfix_dataset_iterate_error_data(do_flatten=do_flatten)
    datasets = [IterateErrorDataSet(pd.DataFrame(dd), vocabulary, name, transformer_vocab_slk=mask_transformer,
                                    do_flatten=do_flatten) for dd, name in
                zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets

    # ac_copy_data_dict = create_copy_addition_data(train_dataset.data_df['ac_code_word_id'],
    #                                               train_dataset.data_df['includes'])
    # ac_copy_dataset = CCodeErrorDataSet(pd.DataFrame(ac_copy_data_dict), vocabulary, 'ac_copy',
    #                                     transformer_vocab_slk=mask_transformer, no_filter=True)
    return train_dataset, valid_dataset, test_dataset, None


def load_deeffix_error_iterative_dataset_real_test(vocabulary, mask_transformer, do_flatten=False):
    data_dict = load_deepfix_error_data_for_iterate()
    test_dataset = IterateErrorDataSet(pd.DataFrame(data_dict), vocabulary, 'deepfix',
                                   transformer_vocab_slk=mask_transformer, do_flatten=do_flatten)
    info_output = "There are {} parsed data in the deepfix dataset".format(len(test_dataset))
    print(info_output)
    return None, None, test_dataset, None

if __name__ == '__main__':
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocab)
    train_dataset, _, _, _ = load_deepfix_sample_iterative_dataset(is_debug=True, vocabulary=vocab,
                                                                   mask_transformer=transformer, do_flatten=True)
    print(len(train_dataset))
