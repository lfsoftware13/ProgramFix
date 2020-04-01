import unittest
from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, \
    load_fake_deepfix_dataset_iterate_error_data_sample_100


class ParseFakeDeepFixDatasetTest(unittest.TestCase):

    def test_not_flatten_testparse(self):
        train_dict, _, _ = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=False)

        # train_dict = {'error_token_id_list': train_data[0], 'sample_error_id_list': train_data[1],
        #               'sample_ac_id_list': train_data[2], 'ac_pos_list': train_data[3],
        #               'error_pos_list': train_data[4], 'includes': train['includes'],
        #               'distance': train['distance']}

        t = 0
        one_error_token_id_list = train_dict['error_token_id_list'].iloc[t]
        one_sample_error_id_list = train_dict['sample_error_id_list'].iloc[t]
        one_sample_ac_id_list = train_dict['sample_ac_id_list'].iloc[t]
        one_ac_pos_list = train_dict['ac_pos_list'].iloc[t]
        one_error_pos_list = train_dict['error_pos_list'].iloc[t]
        one_includes = train_dict['includes'].iloc[t]
        one_distance = train_dict['distance'].iloc[t]
        ac_code_ids = train_dict['ac_code_ids'].iloc[t]

        # for error_id, sample_error, sample_ac, ac_pos, error_pos in \
        #         zip(one_error_token_id_list, one_sample_error_id_list, one_sample_ac_id_list,
        #             one_ac_pos_list, one_error_pos_list):

        for i in range(len(one_error_token_id_list)-1):
            error_id = one_error_token_id_list[i]
            sample_error = one_sample_error_id_list[i]
            sample_ac = one_sample_ac_id_list[i]
            ac_pos = one_ac_pos_list[i]
            error_pos = one_error_pos_list[i]

            output_id = error_id[:error_pos[0]+1] + sample_ac + error_id[error_pos[1]:]
            target_id = one_error_token_id_list[i+1]
            print('for {}'.format(i))
            print('sample_ac: {}'.format(sample_ac))
            print('sample_error: {}'.format(sample_error))
            print(output_id)
            print(target_id)
            self.assertEquals(len(output_id), len(target_id))
            for o, t in zip(output_id, target_id):
                self.assertEquals(o, t)

    def test_flatten_testparse(self):
        train_dict, _, _ = load_fake_deepfix_dataset_iterate_error_data_sample_100(do_flatten=True)

        # train_dict = {'error_token_id_list': train_data[0], 'sample_error_id_list': train_data[1],
        #               'sample_ac_id_list': train_data[2], 'ac_pos_list': train_data[3],
        #               'error_pos_list': train_data[4], 'includes': train['includes'],
        #               'distance': train['distance']}

        i = 0
        one_error_token_id_list = train_dict['error_token_id_list'].iloc[i]
        one_sample_error_id_list = train_dict['sample_error_id_list'].iloc[i]
        one_sample_ac_id_list = train_dict['sample_ac_id_list'].iloc[i]
        one_ac_pos_list = train_dict['ac_pos_list'].iloc[i]
        one_error_pos_list = train_dict['error_pos_list'].iloc[i]
        one_includes = train_dict['includes'].iloc[i]
        one_distance = train_dict['distance'].iloc[i]
        ac_code_ids = train_dict['ac_code_ids'].iloc[i]
        target_ac_token_id_list = train_dict['target_ac_token_id_list'].iloc[i]

        # for error_id, sample_error, sample_ac, ac_pos, error_pos in \
        #         zip(one_error_token_id_list, one_sample_error_id_list, one_sample_ac_id_list,
        #             one_ac_pos_list, one_error_pos_list):

        error_id = one_error_token_id_list
        sample_error = one_sample_error_id_list
        sample_ac = one_sample_ac_id_list
        ac_pos = one_ac_pos_list
        error_pos = one_error_pos_list

        output_id = error_id[:error_pos[0]+1] + sample_ac + error_id[error_pos[1]:]
        target_id = target_ac_token_id_list
        print('sample_ac: {}'.format(sample_ac))
        print('sample_error: {}'.format(sample_error))
        print(output_id)
        print(target_id)
        self.assertEquals(len(output_id), len(target_id))
        for o, t in zip(output_id, target_id):
            self.assertEquals(o, t)

# if __name__ == '__main__':
#     tmp_testparse()
