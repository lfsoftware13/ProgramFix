import torch.nn as nn
import torch
import pandas as pd
import more_itertools

from common import util
from common.problem_util import to_cuda
from common.torch_util import create_sequence_length_mask, MaskOutput
from common.util import CustomerDataSet, PaddedList
from seq2seq.models import EncoderRNN, DecoderRNN
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK


class CCodeDataset(CustomerDataSet):
    def __init__(self, data_df: pd.DataFrame, vocabulary,
                 set_type: str, slk_transformer: TransformVocabularyAndSLK,
                 inner_start_id, inner_end_id, begin_id, end_id,ignore_id=-1, MAX_LENGTH=500):
        self.MAX_LENGTH = MAX_LENGTH
        self.slk_transformer = slk_transformer
        super().__init__(data_df, vocabulary, set_type)
        self.inner_start_id = inner_start_id
        self.inner_end_id = inner_end_id
        self.ignore_id = ignore_id
        self.start_id = begin_id
        self.end_id = end_id

    def filter_df(self, df):
        df = df[df['error_code_word_id'].map(lambda x: x is not None)]
        # print('CCodeErrorDataSet df before: {}'.format(len(df)))
        df = df[df['distance'].map(lambda x: x >= 0)]
        # print('CCodeErrorDataSet df after: {}'.format(len(df)))
        # print(self.data_df['error_code_word_id'])
        # df['error_code_word_id'].map(lambda x: print(type(x)))
        # print(df['error_code_word_id'].map(lambda x: print(x))))
        df = df[df['error_code_word_id'].map(lambda x: len(x) < self.MAX_LENGTH)]
        end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])

        slk_count = 0
        error_count = 0
        not_in_mask_error = 0

        def filter_slk(ac_code_ids):
            res = None
            nonlocal slk_count, error_count, not_in_mask_error
            slk_count += 1
            if slk_count % 100 == 0:
                print('slk count: {}, error count: {}, not in mask error: {}'.format(slk_count, error_count,
                                                                                     not_in_mask_error))
            try:
                mask_list = self.slk_transformer.get_all_token_mask_train([ac_code_ids])[0]
                res = True
                for one_id, one_mask in zip(ac_code_ids + [end_id], mask_list):
                    if one_id not in one_mask:
                        print('id {} not in mask {}'.format(one_id, one_mask))
                        not_in_mask_error += 1
                        res = None
            except Exception as e:
                error_count += 1
                print('slk error : {}'.format(e))
                res = None
            return res

        print('before slk grammar: {}'.format(len(df)))
        df['grammar_mask_list'] = df['ac_code_word_id'].map(filter_slk)
        df = df[df['grammar_mask_list'].map(lambda x: x is not None)]
        print('after slk grammar: {}'.format(len(df)))

        return df

    def _get_raw_sample(self, row):
        res = dict(row)
        ids_set = \
            self.slk_transformer.create_id_constant_string_set_id_by_ids(res['error_code_word_id'])
        slk_parser = self.slk_transformer.create_new_slk_iterator()
        in_sample = False
        compatible_id_list = []
        index_list = []
        pre_list = self.slk_transformer.get_candicate_step(slk_parser, None, ids_set)
        target = []
        for index, target_id in enumerate(res['ac_code_target_id']):
            if target_id == self.inner_start_id:
                in_sample = True
                index_list.append(index)
                compatible_id_list.append(pre_list)
            elif target_id == self.inner_end_id:
                in_sample = False
                pre_list.append(self.inner_end_id)
                target.append(util.get_position(pre_list, target_id))
            else:
                t = self.slk_transformer.get_candicate_step(slk_parser, target_id, ids_set)
                if in_sample:
                    compatible_id_list.append(t)
                    index_list.append(index)
                    target.append(util.get_position(pre_list, target_id))
                pre_list = t
        res['target_index'] = index_list
        res['grammar_index'] = compatible_id_list
        res['target_length'] = len(index_list)
        res['target'] = target
        res['decoder_input'] = [self.start_id] + res['ac_code_target_id'] + [self.end_id]
        res['input_seq'] = res['error_code_word_id'] + [self.end_id]
        res['input_length'] = len(res['input_seq'])
        return res


def create_parse_target_batch_data(ignore_token):
    def parse_target_batch_data(batch_data):
        is_copy = to_cuda(torch.FloatTensor(PaddedList(batch_data['is_copy'], fill_value=ignore_token)))
        target = to_cuda(torch.LongTensor(list(more_itertools.flatten(batch_data['target']))))
        return is_copy, target
    return parse_target_batch_data


def create_loss_fn(ignore_id):
    bce_loss = nn.BCEWithLogitsLoss(reduce=False)
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)

    def loss_fn(is_copy, sample_output, is_copy_target, sample_target):

        is_copy_loss = torch.mean(bce_loss(is_copy.squeeze(-1), is_copy_target)*(is_copy_target != ignore_id).float())
        sample_loss = cross_loss(sample_output, sample_target)
        return is_copy_loss + sample_loss
    return loss_fn


def create_output_ids_fn(model_output, model_input, do_sample):
    if not do_sample:
        fragment_output = model_output[1]
        fragment_output = torch.argmax(fragment_output, dim=1)
        return (model_output[0] > 0.5).float(), fragment_output
    else:
        pass


class Seq2SeqEncoderCopyModel(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 embedding_size,
                 hidden_state_size,
                 start_label,
                 end_label,
                 pad_label,
                 slk_parser,
                 MAX_LENGTH=500,
                 dropout_p=0.1,
                 n_layer=3,):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.sample = False
        self.dropout_p = dropout_p
        self.encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, input_size=embedding_size,
                                  hidden_size=hidden_state_size // 2,
                                  n_layers=n_layer, bidirectional=True, rnn_cell='lstm',
                                  input_dropout_p=self.dropout_p, dropout_p=self.dropout_p, variable_lengths=False,
                                  embedding=None, update_embedding=True)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=MAX_LENGTH, hidden_size=hidden_state_size,
                                  sos_id=start_label, eos_id=end_label, n_layers=n_layer, rnn_cell='lstm',
                                  bidirectional=False, input_dropout_p=self.dropout_p, dropout_p=self.dropout_p,
                                  use_attention=True)
        self.is_copy_output = nn.Linear(hidden_state_size, 1)
        self.grammar_mask_output = MaskOutput(hidden_state_size, vocabulary_size)
        self.decoder_start = torch.ones(1, 1)*start_label
        self.pad_label = pad_label
        self.MAX_LENGTH = MAX_LENGTH
        self.num_layers = n_layer

    def _forward(self, input_seq, input_length, decoder_input,
                 grammar_index, grammar_index_length, target_index):
        encoder_hidden, encoder_mask, encoder_output, is_copy = self._encoder_and_calculate_is_copy(input_length,
                                                                                                    input_seq)
        decoder_output, _, _ = self.decoder(inputs=self.embedding(decoder_input), encoder_hidden=encoder_hidden,
                                            encoder_outputs=encoder_output, encoder_mask=~encoder_mask,
                                            teacher_forcing_ratio=1)
        decoder_output = torch.stack(decoder_output, dim=1)
        max_length = decoder_output.shape[1]
        decoder_output = decoder_output.view(-1, decoder_output.shape[-1])
        to_select_index = []
        for i, index_list in enumerate(target_index):
            for t in index_list:
                to_select_index.append(max_length*i+t)
        decoder_output = torch.index_select(decoder_output, 0, torch.LongTensor(to_select_index).to(input_seq.device))
        grammar_mask = create_sequence_length_mask(grammar_index_length, max_len=grammar_index.shape[1])
        decoder_output = self.grammar_mask_output(decoder_output, grammar_index, grammar_mask)
        return is_copy.squeeze(-1), decoder_output

    def _encoder_and_calculate_is_copy(self, input_length, input_seq):
        input_seq = self.embedding(input_seq)
        encoder_output, encoder_hidden_state = self.encoder(input_seq, )
        is_copy = self.is_copy_output(encoder_output)
        encoder_mask = create_sequence_length_mask(input_length, )
        encoder_hidden = [hid.view(self.num_layers, hid.shape[1], -1) for hid in encoder_hidden_state]
        return encoder_hidden, encoder_mask, encoder_output, is_copy

    def _forward_pre_process(self, batch_data):
        input_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['input_seq'])))
        input_length = to_cuda(torch.LongTensor(batch_data['input_length']))
        decoder_input = to_cuda(torch.LongTensor(PaddedList(batch_data['decoder_input'])))
        grammar_index = list(more_itertools.flatten(batch_data['grammar_index']))
        grammar_index_length = to_cuda(torch.LongTensor([len(t) for t in grammar_index]))
        grammar_index = to_cuda(torch.LongTensor(PaddedList(grammar_index)))
        target_index = batch_data['target_index']
        return input_seq, input_length, decoder_input, grammar_index, grammar_index_length, target_index

    def _forward_sample(self, input_seq, input_length):
        # batch_size = input_seq.size()[0]
        # encoder_hidden, encoder_mask, encoder_output, is_copy = self._encoder_and_calculate_is_copy(input_length,
        #                                                                                             input_seq)
        # is_copy = (is_copy > 0.5).data.cpu().numpy()
        # decoder_start = self.decoder_start.expand(batch_size, -1).to(input_seq.device)
        #
        # continue_mask = to_cuda(torch.Tensor([1 for i in range(batch_size)])).byte()
        # outputs = decoder_start
        # is_copy_stack = []
        # value_output_stack = []
        # pointer_stack = []
        # output_stack = []
        #
        # for i in range(self.MAX_LENGTH):
        #     cur_input_mask = continue_mask.view(continue_mask.shape[0], 1)
        #     output_embed = self.embedding(outputs)
        #
        #     value_output, hidden = self.decoder(inputs=self.embedding(decoder_input), encoder_hidden=encoder_hidden,
        #                                     encoder_outputs=encoder_output, encoder_mask=~encoder_mask,
        #                                     teacher_forcing_ratio=1)
        #     value_output_stack.append(value_output)
        #     pointer_stack.append(pointer_output)
        #
        #     outputs = self.create_next_output(is_copy, value_output, pointer_output, input)
        #     output_stack.append(outputs)
        #
        #     cur_continue = torch.ne(outputs, self.end_label).view(outputs.shape[0])
        #     continue_mask = continue_mask & cur_continue
        #
        #     if torch.sum(continue_mask) == 0:
        #         break
        pass

    def forward(self, batch_data):
        if self.sample:
            pass
        else:
            return self._forward(*self._forward_pre_process(batch_data))


