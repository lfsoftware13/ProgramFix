import os

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

import config
from common import torch_util
from common.args_util import to_cuda
from common.torch_util import create_sequence_length_mask, permute_last_dim_to_second
from common.util import data_loader, show_process_map, PaddedList
from experiment.experiment_util import load_common_error_data, load_common_error_data_sample_100
from model.base_attention import TransformEncoderModel, TransformDecoderModel, TrueMaskedMultiHeaderAttention, \
    register_hook, PositionWiseFeedForwardNet
from read_data.load_data_vocabulary import create_common_error_vocabulary
from vocabulary.word_vocabulary import Vocabulary


MAX_LENGTH = 500
IGNORE_TOKEN = -1
is_debug = True

class CCodeErrorDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transform=None):
        self.data_df = data_df[data_df['error_code_word_id'].map(lambda x: x is not None)]
        # print(self.data_df['error_code_word_id'])
        self.data_df = self.data_df[self.data_df['error_code_word_id'].map(lambda x: len(x) < MAX_LENGTH)]
        self.set_type = set_type
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)
        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        # error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
        #                                                       use_position_label=True)[0]
        # ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
        #                                                       use_position_label=True)[0]

        sample = {"error_tokens": self.data_df.iloc[index]['error_code_word_id'],
                  'error_length': len(self.data_df.iloc[index]['error_code_word_id']),
                  'includes': self.data_df.iloc[index]['includes']}
        if self.set_type != 'valid' or self.set_type != 'test':
            begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[0])
            end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])
            sample['ac_tokens'] = self.data_df.iloc[index]['ac_code_word_id'] + [end_id]
            sample['ac_length'] = len(sample['ac_tokens'])
            sample['token_map'] = self.data_df.iloc[index]['token_map']
            sample['pointer_map'] = create_pointer_map(sample['ac_length'], sample['token_map'])
            sample['error_mask'] = self.data_df.iloc[index]['error_mask']
            sample['is_copy'] = self.data_df.iloc[index]['is_copy'] + [0]
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            sample['token_map'] = None
            sample['pointer_map'] = None
            sample['error_mask'] = None
            sample['is_copy'] = None
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


def create_pointer_map(ac_length, token_map):
    pointer_map = [-1 for i in range(ac_length)]
    for error_i, ac_i in enumerate(token_map):
        if ac_i >= 0:
            pointer_map[ac_i] = error_i
    last_point = -1
    for i in range(len(pointer_map)):
        if pointer_map[i] == -1:
            pointer_map[i] = last_point + 1
        else:
            last_point = pointer_map[i]
            if last_point + 1 >= len(token_map):
                last_point = len(token_map) - 2
    return pointer_map


class SinCosPositionEmbeddingModel(nn.Module):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4):
        super(SinCosPositionEmbeddingModel, self).__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, x, position_start_list=None):
        """

        :param x: has more than 3 dims
        :param position_start_list: len(position_start_list) == len(x.shape) - 2. default: [0] * num_dims.
                create position from start to start+length-1 for each dim.
        :param min_timescale:
        :param max_timescale:
        :return:
        """
        x_shape = list(x.shape)
        num_dims = len(x_shape) - 2
        channels = x_shape[-1]
        num_timescales = channels // (num_dims * 2)
        log_timescales_increment = (math.log(float(self.max_timescale) / float(self.min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = self.min_timescale * to_cuda(torch.exp(torch.range(0, num_timescales - 1) * -log_timescales_increment))
        # add moved position start index
        if position_start_list is None:
            position_start_list = [0] * num_dims
        for dim in range(num_dims):
            length = x_shape[dim + 1]
            # position = transform_to_cuda(torch.range(0, length-1))
            # create position from start to start+length-1 for each dim
            position = to_cuda(torch.range(position_start_list[dim], position_start_list[dim] + length - 1))
            scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad, 0, 0))
            for _ in range(dim + 1):
                signal = torch.unsqueeze(signal, dim=0)
            for _ in range(num_dims - dim - 1):
                signal = torch.unsqueeze(signal, dim=-2)
            x += signal
        return x


class IndexPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(IndexPositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, hidden_size)

    def forward(self, input_index):
        embedded_index = self.embedding(input_index)
        return embedded_index


class SelfAttentionPointerNetworkModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, encoder_stack_num, decoder_stack_num, start_label, end_label, dropout_p=0.1, num_heads=2, normalize_type=None, MAX_LENGTH=500):
        super(SelfAttentionPointerNetworkModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.encoder_stack_num = encoder_stack_num
        self.decoder_stack_num = decoder_stack_num
        self.start_label = start_label
        self.end_label = end_label
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.MAX_LENGTH = MAX_LENGTH

        self.encode_embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.encode_position_embedding = SinCosPositionEmbeddingModel()
        # self.encode_position_embedding = IndexPositionEmbedding(self.hidden_size/2, MAX_LENGTH)
        self.decode_embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.decode_position_embedding = SinCosPositionEmbeddingModel()
        # self.decode_position_embedding = IndexPositionEmbedding(self.hidden_size/2, MAX_LENGTH)

        self.encoder = TransformEncoderModel(hidden_size=self.hidden_size, encoder_stack_num=self.encoder_stack_num, dropout_p=self.dropout_p, num_heads=self.num_heads, normalize_type=self.normalize_type)
        self.decoder = TransformDecoderModel(hidden_size=hidden_size, decoder_stack_num=self.decoder_stack_num, dropout_p=self.dropout_p, num_heads=self.num_heads, normalize_type=self.normalize_type)

        self.copy_linear = nn.Linear(self.hidden_size, 1)

        # self.position_pointer = TrueMaskedMultiHeaderAttention(hidden_size=self.hidden_size, num_heads=1, attention_type='scaled_dot_product')
        # self.position_pointer = nn.Linear(self.hidden_size, self.hidden_size)
        self.position_pointer = PositionWiseFeedForwardNet(self.hidden_size, self.hidden_size, self.hidden_size, 1)
        self.output_linear = nn.Linear(self.hidden_size, self.vocabulary_size)

    def create_next_output(self, copy_output, value_output, pointer_output, input):
        """

        :param copy_output: [batch, sequence, dim**]
        :param value_output: [batch, sequence, dim**, vocabulary_size]
        :param pointer_output: [batch, sequence, dim**, encode_length]
        :param input: [batch, encode_length, dim**]
        :return: [batch, sequence, dim**]
        """
        is_copy = (copy_output > 0.5)
        _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
        _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
        point_id = torch.gather(input, dim=-1, index=pointer_pos_in_input.squeeze(dim=-1))
        next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
        return next_output

    def decode_step(self, decode_input, decode_mask, encode_value, encode_mask):
        decoder_output = self.decoder(decode_input, decode_mask, encode_value, encode_mask)
        is_copy = F.sigmoid(self.copy_linear(decoder_output).squeeze(dim=-1))
        pointer_ff = self.position_pointer(decoder_output)
        pointer_output = torch.bmm(pointer_ff, torch.transpose(encode_value, dim0=-1, dim1=-2))
        if encode_mask is not None:
            dim_len = len(encode_mask.shape)
            pointer_output.masked_fill_(~encode_mask.view(encode_mask.shape[0], *[1 for i in range(dim_len-1)], encode_mask.shape[-1]), -float('inf'))
            # pointer_output = torch.where(torch.unsqueeze(encode_mask, dim=1), pointer_output, to_cuda(torch.Tensor([float('-inf')])))
        value_output = self.output_linear(decoder_output)
        return is_copy, value_output, pointer_output

    def forward(self, input, input_length, output, output_length):
        embedded_input = self.encode_embedding(input)
        position_input = self.encode_position_embedding(embedded_input)
        input_mask = create_sequence_length_mask(input_length)

        encode_value = self.encoder(position_input, input_mask)

        embedded_output = self.decode_embedding(output)
        position_output = self.decode_position_embedding(embedded_output)
        output_mask = create_sequence_length_mask(output_length)

        is_copy, value_output, pointer_output = self.decode_step(position_output, output_mask, encode_value, input_mask)
        return is_copy, value_output, pointer_output

    def forward_test(self, input, input_length):
        embedded_input = self.encode_embedding(input)
        position_input = self.encode_position_embedding(embedded_input)
        input_mask = create_sequence_length_mask(input_length)
        encode_value = self.encoder(position_input, input_mask)

        batch_size = list(input.shape)[0]
        continue_mask = torch.Tensor([1 for i in range(batch_size)]).byte()
        outputs = torch.LongTensor([[self.start_label] for i in range(batch_size)])
        is_copy_stack = []
        value_output_stack = []
        pointer_stack = []
        output_stack = []

        for i in range(self.MAX_LENGTH):
            output_embed = self.decode_embedding(outputs)

            # deal position encode
            dim_num = len(output_embed.shape) - 2
            position_start_list = [i for t in range(dim_num)]
            positioned_output = self.encode_position_embedding(output_embed, position_start_list=position_start_list)

            is_copy, value_output, pointer_output = self.decode_step(positioned_output, decode_mask=None, encode_value=encode_value, encode_mask=input_mask)
            is_copy_stack.append(is_copy)
            value_output_stack.append(value_output)
            pointer_stack.append(pointer_output)

            outputs = self.create_next_output(is_copy, value_output, pointer_output, input)
            output_stack.append(outputs)

            cur_continue = torch.ne(outputs, self.end_label)
            continue_mask = continue_mask & ~cur_continue

            if torch.sum(continue_mask) == 0:
                break

        is_copy_result = torch.cat(is_copy_stack, dim=1)
        value_result = torch.cat(value_output_stack, dim=1)
        pointer_result = torch.cat(pointer_stack, dim=1)
        output_result = torch.cat(output_stack, dim=1)
        return output_result, is_copy_result, value_result, pointer_result


def parse_input_batch_data(batch_data):
    error_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
    error_length = to_cuda(torch.LongTensor(batch_data['error_length']))
    ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
    ac_length = to_cuda(torch.LongTensor(batch_data['ac_length']))
    return error_tokens, error_length, ac_tokens, ac_length


def create_combine_loss_fn():
    copy_loss_fn = nn.BCELoss(reduce=False)
    pointer_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)
    value_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN, reduce=False)

    def combine_total_loss(is_copy, target_copy, pointer_log_probs, target_pointer, value_log_probs, target_value, output_mask):
        output_mask_float = output_mask.float()

        copy_loss = copy_loss_fn(is_copy, target_copy)
        copy_loss = copy_loss * output_mask_float

        pointer_log_probs = permute_last_dim_to_second(pointer_log_probs)
        pointer_loss = pointer_loss_fn(pointer_log_probs, target_pointer)
        pointer_loss = pointer_loss * output_mask_float

        value_log_probs = permute_last_dim_to_second(value_log_probs)
        value_loss = value_loss_fn(value_log_probs, target_value)
        value_loss = value_loss * output_mask_float * (~target_copy.byte()).float()

        # total_loss = copy_loss + pointer_loss + value_loss
        total_loss = pointer_loss
        # register_hook(total_loss, 'total loss')
        return total_loss
    return combine_total_loss


def create_output_ids(is_copy, value_output, pointer_output, error_tokens):
    """

    :param copy_output: [batch, sequence, dim**]
    :param value_output: [batch, sequence, dim**, vocabulary_size]
    :param pointer_output: [batch, sequence, dim**, encode_length]
    :param input: [batch, encode_length, dim**]
    :return: [batch, sequence, dim**]
    """
    is_copy = (is_copy > 0.5)
    _, top_id = torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)
    _, pointer_pos_in_input = torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)
    pointer_pos_in_input = pointer_pos_in_input.squeeze(dim=-1)
    point_id = torch.gather(error_tokens, dim=-1, index=pointer_pos_in_input)
    next_output = torch.where(is_copy, point_id, top_id.squeeze(dim=-1))
    return next_output


def train(model, dataset, batch_size, loss_function, optimizer, clip_norm, epoch_ratio):
    total_loss = to_cuda(torch.Tensor([0]))
    count = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    model.train()

    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=False, drop_last=True, epoch_ratio=epoch_ratio):
        model.zero_grad()
        target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
        target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'])))
        target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
        output_mask = create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['ac_length'])))

        model_input = parse_input_batch_data(batch_data)
        is_copy, value_output, pointer_output = model.forward(*model_input)
        loss = loss_function(is_copy, target_is_copy, pointer_output, target_pointer_output, value_output, target_ac_tokens, output_mask)

        if steps == 0:
            pointer_probs = F.softmax(pointer_output, dim=-1)
            print('pointer output softmax: ', pointer_probs)
            print('pointer output: ', torch.squeeze(torch.topk(pointer_probs, k=1, dim=-1)[1], dim=-1))
            print('target_pointer_output: ', target_pointer_output)

        batch_loss = torch.sum(loss)
        batch_count = torch.sum(output_mask).float()
        loss = batch_loss / batch_count

        # if clip_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        loss.backward()
        optimizer.step()

        output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
        batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
        accuracy = batch_accuracy / batch_count
        total_accuracy += batch_accuracy
        total_loss += batch_loss
        print('in train step {}  loss: {}, accracy: {}'.format(steps, loss.data.item(), accuracy.data.item()))

        count += batch_count
        steps += 1

    return total_loss/count, total_accuracy/count


def evaluate(model, dataset, batch_size, loss_function):
    total_loss = to_cuda(torch.Tensor([0]))
    count = to_cuda(torch.Tensor([0]))
    steps = 0
    total_accuracy = to_cuda(torch.Tensor([0]))
    model.eval()

    with torch.no_grad():
        for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True):
            model.zero_grad()
            target_ac_tokens = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens'])))
            target_pointer_output = to_cuda(torch.LongTensor(PaddedList(batch_data['pointer_map'])))
            target_is_copy = to_cuda(torch.Tensor(PaddedList(batch_data['is_copy'])))
            output_mask = create_sequence_length_mask(to_cuda(torch.LongTensor(batch_data['ac_length'])))

            model_input = parse_input_batch_data(batch_data)
            is_copy, value_output, pointer_output = model.forward(*model_input)
            loss = loss_function(is_copy, target_is_copy, pointer_output, target_pointer_output, value_output,
                                 target_ac_tokens, output_mask)

            batch_loss = torch.sum(loss)
            batch_count = torch.sum(output_mask).float()
            loss = batch_loss / batch_count

            output_ids = create_output_ids(is_copy, value_output, pointer_output, model_input[0])
            batch_accuracy = torch.sum(torch.eq(output_ids, target_ac_tokens) & output_mask).float()
            accuracy = batch_accuracy / batch_count
            total_accuracy += batch_accuracy
            total_loss += batch_loss
            print('in evaluate step {}  loss: {}, accracy: {}'.format(steps, loss.data.item(), accuracy.data.item()))

            count += batch_count
            steps += 1

        # output_result, is_copy, value_output, pointer_output = model.forward_test(*parse_test_batch_data(batch_data))

    return total_loss / count, total_accuracy / count


def train_and_evaluate(batch_size, hidden_size, num_heads, encoder_stack_num, decoder_stack_num, dropout_p, learning_rate, epoches, saved_name, load_name=None, normalize_type='layer', clip_norm=80):
    save_path = os.path.join(config.save_model_root, saved_name)
    if load_name is not None:
        load_path = os.path.join(config.save_model_root, load_name)

    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token, addition_tokens=addition_tokens)

    begin_tokens_id = [vocabulary.word_to_id(i) for i in begin_tokens]
    end_tokens_id = [vocabulary.word_to_id(i) for i in end_tokens]
    unk_token_id = vocabulary.word_to_id(unk_token)
    addition_tokens_id = [vocabulary.word_to_id(i) for i in addition_tokens]

    if is_debug:
        data_dict = load_common_error_data_sample_100()
    else:
        data_dict = load_common_error_data()
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name) for dd, name in
                zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = datasets

    model = SelfAttentionPointerNetworkModel(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
                                     encoder_stack_num=encoder_stack_num, decoder_stack_num=decoder_stack_num,
                                     start_label=begin_tokens_id[0], end_label=end_tokens_id[0], dropout_p=dropout_p,
                                     num_heads=num_heads, normalize_type=normalize_type, MAX_LENGTH=MAX_LENGTH)
    if load_name is not None:
        torch_util.load_model(model, load_path)

    loss_fn = create_combine_loss_fn()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model = to_cuda(model)

    for epoch in range(epoches):
        train_loss, train_accuracy = train(model=model, dataset=train_dataset, batch_size=batch_size, loss_function=loss_fn, optimizer=optimizer,
                               clip_norm=clip_norm, epoch_ratio=1)
        # valid_loss, valid_accuracy = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size, loss_function=loss_fn)
        # test_loss, test_accuracy = evaluate(model=model, dataset=test_dataset, batch_size=batch_size, loss_function=loss_fn)
        scheduler.step(train_loss)
        print('epoch {}: train loss: {}, accuracy: {}'.format(epoch, train_loss, train_accuracy))
        # print('epoch {}: train loss of {}, valid loss of {}, test loss of {}, train_accuracy result of {} '
        #       'valid_accuracy result of {}, test_accuracy result of {}'.format(epoch, train_loss, valid_loss,
        #                                                                        test_loss, train_accuracy,
        #                                                                        valid_accuracy, test_accuracy))
        if not is_debug:
            torch_util.save_model(model, save_path+str(epoch))


if __name__ == '__main__':
    train_and_evaluate(batch_size=2, hidden_size=200, num_heads=1, encoder_stack_num=1, decoder_stack_num=1, dropout_p=0,
                       learning_rate=0.01, epoches=120, saved_name='SelfAttentionPointer.pkl', load_name=None,
                       normalize_type='layer', clip_norm=80)







