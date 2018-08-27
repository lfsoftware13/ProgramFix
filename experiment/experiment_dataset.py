import random

from c_parser.ast_parser import parse_ast_code_graph
from c_parser.pycparser.pycparser import c_ast
from common.pycparser_util import tokenize_by_clex_fn
from common.util import CustomerDataSet, show_process_map, OrderedList
from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data_sample_100, \
    load_fake_deepfix_dataset_iterate_error_data, load_deepfix_error_data_for_iterate, \
    load_deepfix_ac_data_for_generator, load_deepfix_ac_data_for_generator_100, \
    load_generate_code_for_solver_model_iterate_data
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
                 MAX_LENGTH=500,
                 use_ast=False,
                 do_multi_step_sample=False):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        self.is_flatten = do_flatten
        self.max_length = MAX_LENGTH
        self.use_ast = use_ast
        self.transform = False
        # if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':
        #     self.do_sample = False
        # else:
        #     self.do_sample = True
        self.do_multi_step_sample = do_multi_step_sample
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df
            print("before filter p2 out, dataset size is:{}".format(len(self.data_df)))
            self.data_df = self._filter_p2_out(self.data_df)
            print("after filter p2 out, dataset size is:{}".format(len(self.data_df)))
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

        def test_parse_ast_code_graph(seq_name):
            try:
                return parse_ast_code_graph(seq_name[1:-1]).graph_length

            except Exception as e:
                return 710

        if self.use_ast:
            df = df[df['error_token_name_list'].map(test_parse_ast_code_graph) < 700]

        return df

    def _filter_p2_out(self, df):
        def is_no(row):
            try:
                d = self._get_raw_sample(row)
            except Exception as e:
                return False
            if d['copy_length'] <= d['p2_target']:
                return False
            else:
                return True
        return df[df.apply(is_no, raw=True, axis=1)]

    def _get_raw_sample(self, row):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]
        # sample = dict(row)
        sample = {}
        sample['includes'] = row['includes']
        if not self.is_flatten and self.do_multi_step_sample:
            sample['input_seq'] = row['error_token_id_list'][0]
            sample['input_seq_name'] = row['error_token_name_list'][0][1:-1]
            sample['input_length'] = len(sample['input_seq'])
        elif not self.is_flatten and not self.do_multi_step_sample:
            sample['input_seq'] = row['error_token_id_list']
            sample['input_seq_name'] = [r[1:-1] for r in row['error_token_name_list']]
            sample['input_length'] = [len(ids) for ids in sample['input_seq']]
        else:
            sample['input_seq'] = row['error_token_id_list']
            sample['input_seq_name'] = row['error_token_name_list'][1:-1]
            sample['input_length'] = len(sample['input_seq'])
        sample['copy_length'] = sample['input_length']

        inner_begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[1])
        inner_end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[1])
        if not self.do_multi_step_sample:

            # sample['sample_inputs'] = [[inner_begin_id]+one for one in row['sample_ac_id_list']]
            # sample['sample_inputs_length'] = [len(ids) for ids in sample['sample_inputs']]
            if not self.is_flatten:
                sample['is_copy_target'] = [one + [0] for one in row['is_copy_list']]
                sample['copy_target'] = [one + [-1] for one in row['copy_pos_list']]

                sample['sample_target'] = [one + [inner_end_id] for one in row['sample_ac_id_list']]
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

        if self.use_ast:
            code_graph = parse_ast_code_graph(sample['input_seq_name'])
            sample['input_length'] = code_graph.graph_length + 2
            in_seq, graph = code_graph.graph
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['input_seq'] = [begin_id] + [self.vocabulary.word_to_id(t) for t in in_seq] + [end_id]
            sample['adj'] = [[a+1, b+1] for a, b, _ in graph] + [[b+1, a+1] for a, b, _ in graph]

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


class CombineNodeIterateErrorDataSet(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transformer_vocab_slk=None,
                 no_filter=False,
                 do_flatten=False,
                 MAX_LENGTH=500,
                 use_ast=False,
                 do_multi_step_sample=False,
                 id_to_program_dict=None,
                 no_id_to_program_dict=False):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        self.is_flatten = do_flatten
        self.max_length = MAX_LENGTH
        self.use_ast = use_ast
        self.transform = False
        self.id_to_program_dict = id_to_program_dict
        if no_id_to_program_dict:
            self.id_to_program_dict = None
        # if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':
        #     self.do_sample = False
        # else:
        #     self.do_sample = True
        self.do_multi_step_sample = do_multi_step_sample
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df

            if id_to_program_dict is None:
                self.id_to_program_dict = {i: prog_id for i, prog_id in enumerate(sorted(data_df['id']))}
            else:
                self.id_to_program_dict = id_to_program_dict
            if no_id_to_program_dict:
                self.id_to_program_dict = None
            self.only_first = do_multi_step_sample
            self._samples = [FlattenRandomIterateRecords(row, is_flatten=do_flatten, only_first=do_multi_step_sample)
                             for i, row in self.data_df.iterrows()]
            # c = 0
            # for i, (index, row) in self.data_df.iterrows():
            #     print(i)
            #     print(row['id'])
            self.program_to_position_dict = {row['id']: i for i, (index, row) in enumerate(self.data_df.iterrows())}

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

        def test_parse_ast_code_graph(seq_name):
            try:
                return parse_ast_code_graph(seq_name[1:-1]).graph_length

            except Exception as e:
                return 710

        if self.use_ast:
            # df = df[df['error_token_name_list'].map(test_parse_ast_code_graph) < 700]
            pass

        return df

    def set_only_first(self, only_first):
        self.only_first = only_first

    def _get_raw_sample(self, row):
        # sample = dict(row)
        row.select_random_i(only_first=self.only_first)
        sample = {}
        sample['id'] = row['id']
        sample['includes'] = row['includes']
        # if not self.is_flatten and self.do_multi_step_sample:
        #     sample['input_seq'] = row['error_token_id_list'][0]
        #     sample['input_seq_name'] = row['error_token_name_list'][0][1:-1]
        #     sample['input_length'] = len(sample['input_seq'])
        # elif not self.is_flatten and not self.do_multi_step_sample:
        #     sample['input_seq'] = row['error_token_id_list']
        #     sample['input_seq_name'] = [r[1:-1] for r in row['error_token_name_list']]
        #     sample['input_length'] = [len(ids) for ids in sample['input_seq']]
        # else:
        sample['input_seq'] = row['error_token_id_list']
        sample['input_seq_name'] = row['error_token_name_list'][1:-1]
        sample['input_length'] = len(sample['input_seq'])
        sample['copy_length'] = sample['input_length']
        sample['adj'] = 0

        inner_begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[1])
        inner_end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[1])
        if not self.do_multi_step_sample:
            sample['target'] = [inner_begin_id] + row['sample_ac_id_list'] + [inner_end_id]

            sample['is_copy_target'] = row['is_copy_list'] + [0]
            sample['copy_target'] = row['copy_pos_list'] + [-1]

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
            sample['final_output_name'] = row['ac_code_name_with_labels'][1:-1]
            sample['p1_target'] = row['error_pos_list'][0]
            sample['p2_target'] = row['error_pos_list'][1]
            sample['error_pos_list'] = row['error_pos_list']

            sample['distance'] = row['distance']
            sample['includes'] = row['includes']
        else:
            pass

        if self.use_ast:
            code_graph = parse_ast_code_graph(sample['input_seq_name'])
            sample['input_length'] = code_graph.graph_length + 2
            in_seq, graph = code_graph.graph
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['input_seq'] = [begin_id] + [self.vocabulary.word_to_id(t) for t in in_seq] + [end_id]
            sample['adj'] = [[a+1, b+1] for a, b, _ in graph] + [[b+1, a+1] for a, b, _ in graph]

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
        if self.id_to_program_dict is not None:
            prog_id = self.id_to_program_dict[index]
            real_position = self.program_to_position_dict[prog_id]
        else:
            real_position = index
        row = self._samples[real_position]
        return self._get_raw_sample(row)

    def __setitem__(self, key, value):
        if self.id_to_program_dict is not None:
            prog_id = self.id_to_program_dict[key]
            real_position = self.program_to_position_dict[prog_id]
        else:
            real_position = key
        self._samples[real_position] = value

    def __len__(self):
        return len(self._samples)


class CombineDataset:

    def __init__(self, left_dataset, right_dataset, id_to_program_dict:OrderedList, max_dataset_count=1):
        self.left_dataset = left_dataset
        self.right_dataset = right_dataset
        self.dataset_count = max_dataset_count
        self.max_dataset_count = max_dataset_count

        self.id_to_program_dict = id_to_program_dict

    def combine_dataset(self, dataset):
        parent_dataset = CombineDataset(left_dataset=self, right_dataset=dataset,
                                        id_to_program_dict=self.id_to_program_dict,
                                        max_dataset_count=self.max_dataset_count)
        parent_dataset.reduce_child_dataset_count()
        return parent_dataset

    def reduce_child_dataset_count(self):
        if self.left_dataset is not None and isinstance(self.left_dataset, CombineDataset):
            self.left_dataset.reduce_child_dataset_count()
            self.left_dataset.dataset_count -= 1
            if self.left_dataset.dataset_count == 0:
                self.left_dataset = self.left_dataset._merge_datasets_to_one()
        if self.right_dataset is not None and isinstance(self.right_dataset, CombineDataset):
            self.right_dataset.reduce_child_dataset_count()
            self.right_dataset.dataset_count -= 1
            if self.right_dataset.dataset_count == 0:
                self.right_dataset = self.right_dataset._merge_datasets_to_one()

    def _merge_datasets_to_one(self):
        for i in range(len(self)):
            if random.random() > 0.5:
                self.left_dataset[i] = self.right_dataset[i]
        self.right_dataset = None
        return self.left_dataset

    def __getitem__(self, index):
        if random.random() > 0.5 and self.dataset_count > 0:
            return self.right_dataset[index]
        return self.left_dataset[index]

    def __setitem__(self, key, value):
        raise Exception('set item to a combine dataset')

    def __len__(self):
        return len(self.id_to_program_dict)


class FlattenRandomIterateRecords:
    def __init__(self, row, is_flatten, only_first=False):
        self.row = row
        self.is_flatten = is_flatten
        self.multi_layer_keys = ['error_token_id_list', 'sample_error_id_list', 'sample_ac_id_list',
                                 'ac_pos_list', 'error_pos_list', 'is_copy_list', 'copy_pos_list',
                                 'error_token_name_list', 'target_ac_token_id_list']
        if is_flatten:
            self.iterate_num = 1
        else:
            self.iterate_num = len(row['error_token_id_list'])
        self.only_first = only_first
        self.random_i = random.randint(0, self.iterate_num - 1)

    def __getitem__(self, item):
        if self.is_flatten:
            return self.row[item]
        if item in self.multi_layer_keys:
            return self.row[item][self.random_i]
        return self.row[item]

    def select_random_i(self, only_first=None):
        if (only_first is not None and only_first) or \
                (only_first is None and self.only_first):
            self.random_i = 0
        self.random_i = random.randint(0, self.iterate_num - 1)
        # self.random_i = 0


class SamplePackedDataset(CustomerDataSet):

    def __init__(self, dataset, data_len):
        self.datasets = dataset
        self.data_len = data_len
        self.total_len = len(self.datasets)

    def __len__(self):
        if self.total_len < self.data_len:
            return self.total_len
        return self.data_len

    def __getitem__(self, index):
        rand_i = random.randint(0, self.total_len-1)
        return self.datasets[rand_i]

    def add_dataset(self, dataset):
        new_dataset = self.datasets + dataset
        new_packed_dataset = SamplePackedDataset(new_dataset, self.data_len)
        return new_packed_dataset


def load_deepfix_sample_iterative_dataset(is_debug, vocabulary, mask_transformer, do_flatten=False, use_ast=False,
                                          do_multi_step_sample=False, merge_action=True):
    if is_debug:
        data_dict = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=do_flatten, merge_action=merge_action)
    else:
        data_dict = load_fake_deepfix_dataset_iterate_error_data(do_flatten=do_flatten, merge_action=merge_action)
    # if use_ast:
    #     vocabulary = load_graph_vocabulary(vocabulary)

    datasets = [IterateErrorDataSet(pd.DataFrame(dd), vocabulary, name, transformer_vocab_slk=mask_transformer,
                                    do_flatten=do_flatten, use_ast=use_ast, do_multi_step_sample=do_multi_step_sample)
                for dd, name in zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets
    return train_dataset, valid_dataset, test_dataset, None


def load_deepfix_flatten_combine_node_sample_iterative_dataset(is_debug, vocabulary, mask_transformer, do_flatten=False, use_ast=False,
                                          do_multi_step_sample=False, merge_action=True):
    if is_debug:
        data_dict = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=False, merge_action=merge_action)
    else:
        data_dict = load_fake_deepfix_dataset_iterate_error_data(do_flatten=False, merge_action=merge_action)
    # if use_ast:
    #     vocabulary = load_graph_vocabulary(vocabulary)

    datasets = [CombineNodeIterateErrorDataSet(pd.DataFrame(dd), vocabulary, name, transformer_vocab_slk=mask_transformer,
                                    do_flatten=False, use_ast=use_ast, do_multi_step_sample=do_multi_step_sample)
                for dd, name in zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets
    return train_dataset, valid_dataset, test_dataset, None


def load_deepfix_ac_code_for_generate_dataset(is_debug, vocabulary, mask_transformer, do_flatten=False, use_ast=False,
                                          do_multi_step_sample=False):
    if is_debug:
        data_dict = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=False)[0]
    else:
        data_dict = load_fake_deepfix_dataset_iterate_error_data(do_flatten=False)[0]

    def convert_error_to_ac_dict(data_dict):
        new_dict = {}
        new_dict['id'] = data_dict['id']
        new_dict['error_token_id_list'] = data_dict['ac_code_ids']
        new_dict['includes'] = data_dict['includes']
        new_dict['distance'] = data_dict['distance']
        new_dict['error_token_name_list'] = data_dict['ac_code_name_with_labels']
        return new_dict

    ac_data_dict = convert_error_to_ac_dict(data_dict)

    # if use_ast:
    #     vocabulary = load_graph_vocabulary(vocabulary)

    dataset = CombineNodeIterateErrorDataSet(pd.DataFrame(ac_data_dict), vocabulary, 'ac_code',
                                             transformer_vocab_slk=mask_transformer, do_flatten=True,
                                             use_ast=use_ast, do_multi_step_sample=do_multi_step_sample)

    info_output = "There are {} parsed data in the {} dataset".format(len(dataset), 'ac_code')
    print(info_output)
    # info(info_output)
    return dataset


def load_deepfix_ac_code_for_sensibility_rnn(is_debug, vocabulary, mask_transformer, do_flatten=False, use_ast=False,
                                          do_multi_step_sample=False):
    if is_debug:
        data_dict_list = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=False)
    else:
        data_dict_list = load_fake_deepfix_dataset_iterate_error_data(do_flatten=False)

    def convert_error_to_ac_dict(data_dict):
        new_dict = {}
        new_dict['id'] = data_dict['id']
        new_dict['error_token_id_list'] = data_dict['ac_code_ids']
        new_dict['includes'] = data_dict['includes']
        new_dict['distance'] = data_dict['distance']
        new_dict['error_token_name_list'] = data_dict['ac_code_name_with_labels']
        return new_dict

    ac_data_dict_list = [convert_error_to_ac_dict(data_dict) for data_dict in data_dict_list]

    from model.sensibility_baseline.rnn_pytorch import SensibilityRNNDataset
    datasets = [SensibilityRNNDataset(pd.DataFrame(ac_data_dict), vocabulary, 'ac_code',
                                             transformer_vocab_slk=mask_transformer, do_flatten=True,
                                             use_ast=use_ast, do_multi_step_sample=do_multi_step_sample)
               for ac_data_dict in ac_data_dict_list]

    info_output = "There are {} parsed data in the {} dataset".format(len(datasets[0]), 'train')
    print(info_output)
    info_output = "There are {} parsed data in the {} dataset".format(len(datasets[1]), 'valid')
    print(info_output)
    info_output = "There are {} parsed data in the {} dataset".format(len(datasets[2]), 'test')
    print(info_output)
    datasets = datasets + [None]
    return datasets


def load_addition_generate_iterate_solver_train_dataset_fn(vocabulary, mask_transformer, do_flatten=False,
                                                        use_ast=False, do_multi_step_sample=False):
    def load_addition_generate_iterate_solver_train_dataset(df, id_to_prog_dict, no_id_to_program_dict=False):
        df_dict = load_generate_code_for_solver_model_iterate_data(df, convert_field_fn=None, convert_field_dict={},
                                                         do_flatten=False, vocabulary=vocabulary)
        addition_dataset = CombineNodeIterateErrorDataSet(pd.DataFrame(df_dict), vocabulary, 'generate',
                                               transformer_vocab_slk=mask_transformer, do_flatten=False,
                                               use_ast=use_ast, do_multi_step_sample=do_multi_step_sample,
                                                          id_to_program_dict=id_to_prog_dict,
                                                          no_id_to_program_dict=no_id_to_program_dict)
        return addition_dataset
    return load_addition_generate_iterate_solver_train_dataset


def load_graph_vocabulary(vocabulary):
    vocabulary.add_token("<Delimiter>")
    ast_node_dict = c_ast.__dict__
    for n in sorted(ast_node_dict):
        s_c = ast_node_dict[n]
        b_c = c_ast.Node
        try:
            if issubclass(s_c, b_c):
                vocabulary.add_token(n)
        except Exception as e:
            pass
    return vocabulary


def load_deeffix_error_iterative_dataset_real_test(vocabulary, mask_transformer, do_flatten=False, use_ast=False,
                                                   do_multi_step_sample=True):
    data_dict = load_deepfix_error_data_for_iterate()
    test_dataset = IterateErrorDataSet(pd.DataFrame(data_dict), vocabulary, 'deepfix',
                                   transformer_vocab_slk=mask_transformer, do_flatten=do_flatten, use_ast=use_ast,
                                       do_multi_step_sample=do_multi_step_sample)
    info_output = "There are {} parsed data in the deepfix dataset".format(len(test_dataset))
    print(info_output)
    return None, None, test_dataset, None

if __name__ == '__main__':
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(tokenize_fn=tokenize_fn, vocab=vocab)
    train_dataset = load_deepfix_ac_code_for_generate_dataset(is_debug=True, vocabulary=vocab,
                                                              mask_transformer=transformer,
                                                              do_flatten=True, use_ast=False)
    print(len(train_dataset))
