import itertools
import math
import os
import sys
import time

import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd

from torch.distributions import Categorical
from torch.utils.data import Dataset

import config
from common import torch_util
from common.torch_util import create_sequence_length_mask, calculate_accuracy_of_code_completion
from common.util import data_loader, PaddedList, show_process_map
from experiment.experiment_util import load_common_error_data_sample_100, load_common_error_data
from experiment.parse_xy_util import combine_spilt_tokens_batch, check_action_include_all_error, \
    combine_spilt_tokens_batch_with_tensor
from model.model_test.only_attention_fix_error_model import OnlyAttentionFixErrorModelWithoutInputEmbedding
from read_data.load_data_vocabulary import create_common_error_vocabulary
from vocabulary.word_vocabulary import Vocabulary

is_cuda = True
gpu_index = 1
is_debug = False
MAX_LENGTH = 500
TARGET_PAD_TOKEN = -1


def transform_to_cuda(x):
    if is_cuda:
        x = x.cuda(gpu_index)
    return x


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
            sample['ac_tokens'] = self.data_df.iloc[index]['ac_code_word_id']
            sample['ac_length'] = len(self.data_df.iloc[index]['ac_code_word_id'])
            sample['token_map'] = self.data_df.iloc[index]['token_map']
            sample['error_mask'] = self.data_df.iloc[index]['error_mask']
        else:
            sample['ac_tokens'] = None
            sample['ac_length'] = 0
            sample['token_map'] = None
            sample['error_mask'] = None
        return sample



    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


class SelectPolicy(nn.Module):
    def __init__(self, input_size, action_num):
        super(SelectPolicy, self).__init__()
        self.input_size = input_size
        self.action_num = action_num
        self.affine = transform_to_cuda(nn.Linear(input_size, action_num))

    def forward(self, inputs):
        out = self.affine(inputs)
        out = F.sigmoid(out)
        return out


class StructedRepresentationRNN(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_layers, batch_size, dropout_p=0.1):
        super(StructedRepresentationRNN, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout_p = dropout_p

        self.context_hidden_size = hidden_size
        self.word_hidden_size = self.context_hidden_size * 2

        self.embedding = transform_to_cuda(nn.Embedding(vocabulary_size, hidden_size))

        self.context_rnn = transform_to_cuda(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True, bidirectional=True))

        self.word_rnn = transform_to_cuda(nn.LSTM(input_size=self.word_hidden_size, hidden_size=self.word_hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True))

        self.begin_token = transform_to_cuda(nn.Parameter(torch.randn(self.word_hidden_size)))
        self.end_token = transform_to_cuda(nn.Parameter(torch.randn(self.word_hidden_size)))
        self.gap_token = transform_to_cuda(nn.Parameter(torch.randn(self.word_hidden_size)))


    def init_context_hidden(self):
        bidirectional_num = 2
        hidden = (torch.autograd.Variable(torch.randn(self.num_layers * bidirectional_num, self.batch_size, self.context_hidden_size)),
                  torch.autograd.Variable(torch.randn(self.num_layers * bidirectional_num, self.batch_size, self.context_hidden_size)))
        hidden = [transform_to_cuda(x) for x in hidden]
        return hidden

    def init_hidden(self):
        bidirectional_num = 1
        hidden = (torch.autograd.Variable(torch.randn(self.num_layers * bidirectional_num, self.batch_size, self.word_hidden_size)),
                  torch.autograd.Variable(torch.randn(self.num_layers * bidirectional_num, self.batch_size, self.word_hidden_size)))
        hidden = [transform_to_cuda(x) for x in hidden]
        return hidden

    def forward_one_step(self, inputs):
        output, self.hidden = self.word_rnn(inputs, self.hidden)
        return output, self.hidden

    def do_context_rnn(self, inputs):
        embedding_input = self.embedding(inputs)

        self.context_hidden = self.init_context_hidden()
        context_input, self.context_hidden = self.context_rnn(embedding_input, self.context_hidden)
        return context_input, self.context_hidden


def combine_train(p_model, s_model, seq_model, dataset, batch_size, loss_fn, p_optimizer, s_optimizer, delay_reward_fn, baseline_fn, delay_loss_fn, vocab, train_type=None, predict_type='first', include_error_reward=-10000, pretrain=False, random_action=None):
    if train_type == 'p_model':
        change_model_state([p_model], [s_model, seq_model])
        policy_train = True
    elif train_type == 's_model':
        change_model_state([s_model, seq_model], [p_model])
        policy_train = False
    else:
        change_model_state([], [p_model, s_model, seq_model])
        policy_train = False

    begin_tensor = s_model.begin_token
    end_tensor = s_model.end_token
    gap_tensor = s_model.gap_token

    begin_len = 1
    begin_token = vocab.word_to_id(vocab.begin_tokens[0])
    end_token = vocab.word_to_id(vocab.end_tokens[0])
    gap_token = vocab.word_to_id(vocab.addition_tokens[0])
    step = 0
    select_count = torch.LongTensor([0])
    seq_count = torch.LongTensor([0])
    decoder_input_count = torch.LongTensor([0])
    total_seq_loss = torch.Tensor([0])
    total_p_loss = torch.Tensor([0])
    total_s_accuracy_top_k = {}
    for data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        p_model.zero_grad()
        s_model.zero_grad()
        seq_model.zero_grad()

        error_tokens = transform_to_cuda(torch.LongTensor(PaddedList(data['error_tokens'])))
        error_length = transform_to_cuda(torch.LongTensor(data['error_length']))
        error_action_masks = transform_to_cuda(torch.ByteTensor(PaddedList(data['error_mask'], fill_value=0)))

        max_len = torch.max(error_length)
        error_token_masks = create_sequence_length_mask(error_length, max_len=max_len.data.item(), gpu_index=gpu_index)

        # add full code context information to each position word using BiRNN.
        context_input, context_hidden = s_model.do_context_rnn(error_tokens)
        # sample the action by interaction between policy model(p_model) and structed model(s_model)
        if not pretrain:
            action_probs_records_list, action_records_list, output_records_list, hidden = create_policy_action_batch(p_model, s_model, context_input, policy_train=policy_train)
        else:
            action_probs_records_list, action_records_list, output_records_list, hidden = create_policy_action_batch(p_model, s_model, context_input, policy_train=True, random_action=[0.8, 0.2])
        action_probs_records = torch.stack(action_probs_records_list, dim=1)
        action_records = torch.stack(action_records_list, dim=1)
        output_records = torch.cat(output_records_list, dim=1)
        masked_action_records = action_records.data.masked_fill_(~error_token_masks, 0)
        if pretrain:
            masked_action_records = error_action_masks.byte() | masked_action_records.byte()

        include_all_error = check_action_include_all_error(masked_action_records, error_action_masks)
        contain_all_error_count = torch.sum(include_all_error)

        tokens_tensor, token_length, part_ac_tokens_list, ac_token_length = combine_spilt_tokens_batch_with_tensor(
            output_records, data['ac_tokens'], masked_action_records, data['token_map'], gap_tensor, begin_tensor,
            end_tensor, gap_token, begin_token, end_token, gpu_index=gpu_index)

        if predict_type == 'start':
            decoder_input = [tokens[:-1] for tokens in part_ac_tokens_list]
            decoder_length = [len(inp) for inp in decoder_input]
            target_output = [tokens[1:] for tokens in part_ac_tokens_list]
        elif predict_type == 'first':
            decoder_input = [tokens[begin_len:-1] for tokens in part_ac_tokens_list]
            decoder_length = [len(inp) for inp in decoder_input]
            target_output = [tokens[begin_len+1:] for tokens in part_ac_tokens_list]

        token_length_tensor = transform_to_cuda(torch.LongTensor(token_length))
        ac_token_tensor = transform_to_cuda(torch.LongTensor(PaddedList(decoder_input, fill_value=0)))
        ac_token_length_tensor = transform_to_cuda(torch.LongTensor(decoder_length))
        log_probs = seq_model.forward(tokens_tensor, token_length_tensor, ac_token_tensor, ac_token_length_tensor)

        target_output_tensor = transform_to_cuda(torch.LongTensor(PaddedList(target_output, fill_value=TARGET_PAD_TOKEN)))
        s_loss = loss_fn(log_probs.view(-1, vocab.vocabulary_size), target_output_tensor.view(-1))

        remain_batch = torch.sum(masked_action_records, dim=1)
        add_batch = torch.eq(remain_batch, 0).long()
        remain_batch = remain_batch + add_batch
        total_batch = torch.sum(error_token_masks, dim=1)
        force_error_rewards = (~include_all_error).float() * include_error_reward
        delay_reward = delay_reward_fn(log_probs, target_output_tensor, total_batch, remain_batch, force_error_rewards)
        delay_reward = torch.unsqueeze(delay_reward, dim=1).expand(-1, max_len)
        delay_reward = delay_reward * error_token_masks.float()

        if baseline_fn is not None:
            baseline_reward = baseline_fn(delay_reward, error_token_masks)
            total_reward = delay_reward - baseline_reward
        else:
            total_reward = delay_reward

        # force_error_rewards = torch.unsqueeze(~include_all_error, dim=1).float() * error_token_masks.float() * include_error_reward
        force_error_rewards = torch.unsqueeze(~include_all_error, dim=1).float() * error_token_masks.float() * 0
        p_loss = delay_loss_fn(action_probs_records, total_reward, error_token_masks, force_error_rewards)

        if math.isnan(p_loss):
            print('p_loss is nan')
            continue
        # iterate record variable
        step += 1
        one_decoder_input_count = torch.sum(ac_token_length_tensor)
        decoder_input_count += one_decoder_input_count.data.cpu()
        total_seq_loss += s_loss.cpu().data.item() * one_decoder_input_count.float().cpu()

        one_seq_count = torch.sum(error_length)
        seq_count += one_seq_count.cpu()
        total_p_loss += p_loss.cpu().data.item() * one_seq_count.float().cpu()

        s_accuracy_top_k = calculate_accuracy_of_code_completion(log_probs, target_output_tensor, ignore_token=TARGET_PAD_TOKEN, topk_range=(1, 5), gpu_index=gpu_index)
        for key, value in s_accuracy_top_k.items():
            total_s_accuracy_top_k[key] = s_accuracy_top_k.get(key, 0) + value

        select_count_each_batch = torch.sum(masked_action_records, dim=1)
        select_count = select_count + torch.sum(select_count_each_batch).data.cpu()

        print(
            'train_type: {} step {} sequence model loss: {}, policy model loss: {}, contain all error count: {}, select of each batch: {}, total of each batch: {}, total decoder_input_cout: {}, topk: {}, '.format(
                train_type, step, s_loss, p_loss, contain_all_error_count, select_count_each_batch.data.tolist(), error_length.data.tolist(), one_decoder_input_count.data.item(), s_accuracy_top_k))
        sys.stdout.flush()
        sys.stderr.flush()

        if train_type != 'p_model':
            p_model.zero_grad()
        if train_type != 's_model':
            s_model.zero_grad()
            seq_model.zero_grad()

        if train_type == 'p_model':
            torch.nn.utils.clip_grad_norm_(p_model.parameters(), 0.5)
            p_loss.backward()
            p_optimizer.step()
        elif train_type == 's_model':
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), 8)
            torch.nn.utils.clip_grad_norm_(seq_model.parameters(), 8)
            s_loss.backward()
            s_optimizer.step()

    for key, value in total_s_accuracy_top_k.items():
        total_s_accuracy_top_k[key] = total_s_accuracy_top_k.get(key, 0) / decoder_input_count.data.item()

    return (total_seq_loss/decoder_input_count.float()).data.item(), (total_p_loss/seq_count.float()).data.item(), (select_count.float()/seq_count.float()).data.item(), total_s_accuracy_top_k


def train_and_evaluate(data_type, batch_size, hidden_size, num_heads, encoder_stack_num, decoder_stack_num, structed_num_layers, addition_reward_gamma, baseline_min_len, length_punish_scale, dropout_p, learning_rate, epoches, saved_name, load_name=None, gcc_file_path='test.c', normalize_type='layer', predict_type='start', pretrain_s_model_epoch=0):
    save_path = os.path.join(config.save_model_root, saved_name)
    if load_name is not None:
        load_path = os.path.join(config.save_model_root, load_name)

    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                                                addition_tokens=addition_tokens)

    begin_tokens_id = [vocabulary.word_to_id(i) for i in begin_tokens]
    end_tokens_id = [vocabulary.word_to_id(i) for i in end_tokens]
    unk_token_id = vocabulary.word_to_id(unk_token)
    addition_tokens_id = [vocabulary.word_to_id(i) for i in addition_tokens]

    if is_debug:
        data_dict = load_common_error_data_sample_100()
    else:
        data_dict = load_common_error_data()
    datasets = [CCodeErrorDataSet(pd.DataFrame(dd), vocabulary, name) for dd, name in zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = datasets

    seq_model = OnlyAttentionFixErrorModelWithoutInputEmbedding(vocabulary_size=vocabulary.vocabulary_size, hidden_size=2*hidden_size,
                                       sequence_max_length=MAX_LENGTH, num_heads=num_heads,
                                       start_label=vocabulary.word_to_id(vocabulary.begin_tokens[0]),
                                       end_label=vocabulary.word_to_id(vocabulary.end_tokens[0]), pad_label=0,
                                       encoder_stack_num=encoder_stack_num, decoder_stack_num=decoder_stack_num,
                                       dropout_p=dropout_p, normalize_type=normalize_type)
    p_model = SelectPolicy(input_size=2*hidden_size+2*hidden_size+2*hidden_size, action_num=2)
    s_model = StructedRepresentationRNN(vocabulary_size=vocabulary.vocabulary_size, hidden_size=hidden_size,
                                        num_layers=structed_num_layers, batch_size=batch_size, dropout_p=dropout_p)

    seq_loss = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_TOKEN)
    seq_loss_no_reduce = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_TOKEN, reduce=False)

    p_optimizer = torch.optim.SGD(p_model.parameters(), lr=learning_rate)
    p_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(p_optimizer, 'min')
    s_optimizer = torch.optim.SGD(itertools.chain(s_model.parameters(), seq_model.parameters()), lr=learning_rate)
    s_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(p_optimizer, 'min')
    best_p_valid_loss = None
    best_p_test_loss = None
    best_s_valid_loss = None
    best_s_test_loss = None

    remain_length_punish = create_remain_punish(length_punish_scale)
    # delay_reward_fn = create_delayed_reward_fn(seq_loss_no_reduce, gamma=addition_reward_gamma, length_punish_fn=delete_length_punish)
    delay_reward_fn = create_delayed_reward_fn(seq_loss_no_reduce, gamma=addition_reward_gamma, length_punish_fn=remain_length_punish)
    # baseline_fn = create_baseline_fn(baseline_min_len)
    baseline_fn = None
    delay_loss_fn = create_delay_loss_fn(do_normalize=False)

    if load_name is not None:
        torch_util.load_model(p_model, load_path+'_p', map_location={'cuda:1':'cuda:0'})
        torch_util.load_model(s_model, load_path+'_s', map_location={'cuda:1':'cuda:0'})
        torch_util.load_model(seq_model, load_path+'_seq', map_location={'cuda:1':'cuda:0'})
        best_s_valid_loss, best_p_valid_loss, valid_select_probs, valid_accuracy_topk = combine_train(p_model, s_model, seq_model, valid_dataset, batch_size,
                                                     loss_fn=seq_loss, p_optimizer=p_optimizer, s_optimizer=s_optimizer,
                                                     delay_reward_fn=delay_reward_fn, baseline_fn=baseline_fn,
                                                     delay_loss_fn=delay_loss_fn, vocab=vocabulary, train_type='valid',
                                                     predict_type='first', include_error_reward=-10000, pretrain=False)
        best_s_test_loss, best_p_test_loss, test_select_probs, test_accuracy_topk = combine_train(p_model, s_model, seq_model, test_dataset, batch_size,
                                                   loss_fn=seq_loss, p_optimizer=p_optimizer, s_optimizer=s_optimizer,
                                                   delay_reward_fn=delay_reward_fn, baseline_fn=baseline_fn,
                                                   delay_loss_fn=delay_loss_fn, vocab=vocabulary, train_type='test',
                                                   predict_type='first', include_error_reward=-10000, pretrain=False)

    if pretrain_s_model_epoch > 0:
        for i in range(pretrain_s_model_epoch):
            print('in epoch: {}, train type: {}'.format(i, 'pre-train'))
            pretrain_seq_loss, pretrain_p_loss, pretrain_select_probs, pretrain_accuracy_topk = combine_train(p_model, s_model, seq_model, train_dataset, batch_size,
                                                         loss_fn=seq_loss, p_optimizer=p_optimizer, s_optimizer=s_optimizer,
                                                         delay_reward_fn=delay_reward_fn, baseline_fn=baseline_fn,
                                                         delay_loss_fn=delay_loss_fn, vocab=vocabulary,
                                                         train_type='s_model', predict_type='first',
                                                         include_error_reward=-10000, pretrain=True, random_action=[0.6, 0.4])
            print('pretrain {}: seq_loss: {}, p_loss: {}'.format(str(i), pretrain_seq_loss, pretrain_p_loss))

    for epoch in range(epoches):
        if int(epoch/2) % 2 == 0:
            train_type = 'p_model'
        else:
            train_type = 's_model'
            # train_type = 'p_model'
        print('in epoch: {}, train type: {}'.format(epoch, train_type))
        train_seq_loss, train_p_loss, train_select_probs, train_accuracy_topk = combine_train(p_model, s_model, seq_model, train_dataset, batch_size,
                                                     loss_fn=seq_loss, p_optimizer=p_optimizer, s_optimizer=s_optimizer,
                                                     delay_reward_fn=delay_reward_fn, baseline_fn=baseline_fn,
                                                     delay_loss_fn=delay_loss_fn, vocab=vocabulary,
                                                     train_type=train_type, predict_type='first',
                                                     include_error_reward=-10000, pretrain=False)
        if not is_debug:
            valid_seq_loss, valid_p_loss, valid_select_probs, valid_accuracy_topk = combine_train(p_model, s_model, seq_model, valid_dataset, batch_size,
                                                         loss_fn=seq_loss, p_optimizer=p_optimizer, s_optimizer=s_optimizer,
                                                         delay_reward_fn=delay_reward_fn, baseline_fn=baseline_fn,
                                                         delay_loss_fn=delay_loss_fn, vocab=vocabulary, train_type='valid',
                                                         predict_type='first', include_error_reward=-10000, pretrain=False)
            test_seq_loss, test_p_loss, test_select_probs, test_accuracy_topk = combine_train(p_model, s_model, seq_model, test_dataset, batch_size,
                                                       loss_fn=seq_loss, p_optimizer=p_optimizer, s_optimizer=s_optimizer,
                                                       delay_reward_fn=delay_reward_fn, baseline_fn=baseline_fn,
                                                       delay_loss_fn=delay_loss_fn, vocab=vocabulary, train_type='test',
                                                       predict_type='first', include_error_reward=-10000, pretrain=False)
        else:
            valid_seq_loss = 0
            valid_p_loss = 0
            valid_select_probs = 0
            valid_accuracy_topk = {}
            test_seq_loss = 0
            test_p_loss = 0
            test_select_probs = 0
            test_accuracy_topk = {}
        # train_seq_loss = 0
        # train_p_loss = 0
        # valid_seq_loss = 0
        # valid_p_loss = 0
        # test_seq_loss = 0
        # test_p_loss = 0

        if train_type == 's_model':
            s_scheduler.step(valid_seq_loss)
        elif train_type == 'p_model':
            p_scheduler.step(valid_p_loss)

        if best_s_valid_loss is None or valid_seq_loss < best_s_valid_loss:
            best_p_valid_loss = valid_p_loss
            best_p_test_loss = test_p_loss
            best_s_valid_loss = valid_seq_loss
            best_s_test_loss = test_seq_loss
        if not is_debug:
            cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            torch_util.save_model(p_model, '{}_{}_{}_{}'.format(save_path, 'p', str(epoch), cur_time))
            torch_util.save_model(s_model, '{}_{}_{}_{}'.format(save_path, 's', str(epoch), cur_time))
            torch_util.save_model(seq_model, '{}_{}_{}_{}'.format(save_path, 'seq', str(epoch), cur_time))
        print('epoch {}: train_seq_loss: {}, train_p_loss: {}, train_select_probs: {}, train_accuracy: {}, '
              'valid_seq_loss: {}, valid_p_loss: {}, valid_select_probs: {}, valid_accuracy: {}, '
              'test_seq_loss: {}, test_p_loss: {}, test_select_probs: {}, test_accuracy: {}'
              .format(epoch, train_seq_loss, train_p_loss, train_select_probs, train_accuracy_topk,
                      valid_seq_loss, valid_p_loss, valid_select_probs, valid_accuracy_topk,
                      test_seq_loss, test_p_loss, test_select_probs, test_accuracy_topk))
    print('the model {} best valid_seq_loss: {}, best valid_p_loss: {}, '
          'best test_seq_loss: {}, best test_p_loss: {}'.format(saved_name, best_s_valid_loss, best_p_valid_loss,
                                                                best_s_test_loss, best_p_test_loss))


# --------------------------- two kinds of loss_fn ------------------------ #
def create_delayed_reward_fn(loss_fn, gamma, length_punish_fn):

    def calculate_delayed_rewards(log_probs, target_token_tensor, total, remain, force_reward):
        """

        :param log_probs:
        :param target_token_tensor:
        :param total: [batch]
        :param remain: [batch]
        :param force_reward : [batch]
        :return:
        """
        total = total.float()
        remain = remain.float()
        batch_size = log_probs.shape[0]
        length = torch.sum(torch.ne(target_token_tensor, -1).view(batch_size, -1), dim=1)
        loss_full_shape = loss_fn(torch.transpose(log_probs, dim0=1, dim1=2), target_token_tensor)
        loss_batch = torch.sum(loss_full_shape.view(batch_size, -1), dim=1).float()/length.float()
        length_punish_reward = length_punish_fn(total, remain) * gamma
        total_reward_batch = - loss_batch + length_punish_reward + force_reward
        if math.isnan(total_reward_batch.view(-1).data[0]):
            print('is nan')
        return total_reward_batch

        # origin_rewards = []
        # R = 0
        # policy_loss = []
        # rewards = []
        # for r in policy.rewards[::-1]:
        #     R = r + reinforce_gamma * R
        #     rewards.insert(0, R)
        # rewards = torch.tensor(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        # for log_prob, reward in zip(policy.saved_log_probs, rewards):
        #     policy_loss.append(-log_prob * reward)
        # optimizer.zero_grad()
        # policy_loss = torch.cat(policy_loss).sum()

    return calculate_delayed_rewards


def create_baseline_fn(baseline_min_len):
    def step_mean_baseline(delay_rewards, error_mask):
        batch_size = error_mask.shape[0]
        step_len = torch.sum(error_mask, dim=0).float()
        no_baseline_mask = step_len < baseline_min_len
        step_len = step_len.data.masked_fill_(no_baseline_mask, float('inf'))
        last_effect_baseline_index = torch.sum(~no_baseline_mask) - 1

        # full_rewards_tensor = torch.unsqueeze(delay_rewards, dim=1) * error_mask.float()
        baseline_rewards_one_seq = torch.sum(delay_rewards, dim=0) / step_len
        last_effect_baseline = baseline_rewards_one_seq[last_effect_baseline_index]
        baseline_rewards_one_seq = baseline_rewards_one_seq + last_effect_baseline * no_baseline_mask.float()
        expanded_baseline_rewards = torch.unsqueeze(baseline_rewards_one_seq, dim=0).expand(batch_size, -1)
        baseline_rewards = expanded_baseline_rewards * error_mask.float()
        return baseline_rewards
    return step_mean_baseline


def create_delay_loss_fn(do_normalize=True):
    def delay_loss_with_force_mask(action_log_probs, total_reward, error_mask, force_value):
        """
        :param action_log_probs: [batch, seq_len]
        :param total_reward: [batch, seq_len]
        :param error_mask: [batch, seq_len]
        :param force_value: [batch, seq_len], add force value to final reward
        :return:
        """
        if do_normalize:
            norm_reward = normalize_rewards(total_reward, error_mask)
        else:
            norm_reward = total_reward
        norm_reward = norm_reward + force_value

        # loss_position shape: [batch, seq_len]. make sure norm_reward.data.required_grad == False
        loss_position = - action_log_probs * norm_reward.data

        loss = torch.mean(torch.masked_select(loss_position, error_mask))
        return loss

    return delay_loss_with_force_mask


def normalize_rewards(rewards, error_mask):
    # effect_rewards shape: [-1]
    effect_rewards = torch.masked_select(rewards, error_mask)
    mean = torch.mean(effect_rewards)
    std = torch.std(effect_rewards)

    norm_reward = (rewards - mean) / std
    return norm_reward


def delete_length_punish(total, remain):
    return (total-remain)/total


def create_remain_punish(scale):
    def remain_reward_fn(total, remain):
        """
        it will get the maximun reward when remain/total = sqrt(scale)
        :param total:
        :param remain:
        :return:
        """
        reward = remain / total + scale * total / remain
        return -reward
    return remain_reward_fn


# ---------------------------------------- model calculate util method ------------------------------------ #
def create_policy_action_batch(p_model, s_model, inputs, hidden=None, policy_train=False, random_action=None):
    """
    create action step by step using a policy net to produce action and a rnn to produce state(last hidden state with
    current input).
    :param p_model: policy net
    :param s_model: structed net. it contains a rnn and produce next state to policy. (as the environment)
    :param inputs: [batch, seq_len, hidden_size]. A batch data
    :param hidden: last context hidden state. init hidden as zeros if hidden is None.
    :param policy_train: random action in policy if policy_train is True else choose the max probs action
    :param random_action: a python list of action probility. such as [0.8, 0.2]. it means action 0 will be chosen in 80% probs
                            and action 1 will be chosen in 20% probs.
    :return:
    """
    if hidden is None:
        hidden = s_model.init_hidden()
    s_model.hidden = hidden
    iteration_times = list(inputs.shape)[1]
    batch_size = list(inputs.shape)[0]
    action_probs_records = []
    action_records = []
    output_records = []

    for i in range(iteration_times):
        cur_inp = torch.unsqueeze(inputs[:, i, :], dim=1)
        context_state = concat_hidden_to_state_batch(hidden, cur_inp)
        if random_action is None:
            action_value = p_model.forward(context_state.data).view(batch_size, p_model.action_num)
        else:
            action_value = random_action_value_with_probs(batch_size, random_action)

        action, action_probs = sample_action(action_value, policy_train)
        action_records = action_records + [action]
        action_probs_records = action_probs_records + [action_probs]

        s_model.hidden = hidden
        one_out, one_hidden = s_model.forward_one_step(cur_inp)
        output_records = output_records + [one_out]

        hidden = [select_hidden_by_action_batch(action, one_hid, last_hid) for one_hid, last_hid in zip(one_hidden, hidden)]

    return action_probs_records, action_records, output_records, hidden


def sample_action(action_value, policy_train):
    m = Categorical(action_value)
    if policy_train:
        action = m.sample()
    else:
        action = torch.topk(action_value, k=1, dim=1)[1].view(-1)
    action_probs = m.log_prob(action)
    return transform_to_cuda(action), transform_to_cuda(action_probs)


def select_hidden_by_action_batch(action, one_hidden, last_hidden):
    batch_first_one_hidden = torch.transpose(one_hidden, dim0=0, dim1=1)
    batch_first_last_hidden = torch.transpose(last_hidden, dim0=0, dim1=1)
    unsqueeze_action = action
    for i in range(len(batch_first_one_hidden.shape)-len(action.shape)):
        unsqueeze_action = torch.unsqueeze(unsqueeze_action, dim=-1)
    batch_first_masked_one_hidden = torch.where(unsqueeze_action.byte(), batch_first_one_hidden, batch_first_last_hidden)
    return torch.transpose(batch_first_masked_one_hidden, dim0=0, dim1=1)


def concat_hidden_to_state_batch(hidden, inp):

    def concat_rnn_hidden_dim_batch(one_hidden):
        """
        :param one_hidden: [num_layers * bidirectional_num, batch_size, hidden_size]. rnn hidden tensor or rnn cell tensor.
        :return: [batch_size, num_layers * bidirectional_num * hidden_size]. concat the last dim.of hidden.
        """
        one_hidden = torch.transpose(one_hidden, dim0=0, dim1=1)
        one_hidden = one_hidden.view(one_hidden.shape[0], -1)
        return one_hidden

    hidden_state = torch.unsqueeze(concat_rnn_hidden_dim_batch(hidden[0]), dim=1)
    cell_state = torch.unsqueeze(concat_rnn_hidden_dim_batch(hidden[1]), dim=1)
    concat_context_state = torch.cat([inp, hidden_state, cell_state], dim=2)
    return concat_context_state


def change_model_state(train_models, evaluate_models):
    for model in train_models:
        model.train()
    for model in evaluate_models:
        model.eval()


def random_action_value_with_probs(batch_size, action_probs):
    one_action = torch.Tensor(action_probs)
    actions = torch.unsqueeze(one_action, dim=0)
    actions = actions.expand(batch_size, -1)
    return actions


if __name__ == '__main__':

    train_and_evaluate('common_error', batch_size=8, hidden_size=128, num_heads=4, encoder_stack_num=5,
                       decoder_stack_num=5,
                       structed_num_layers=1, addition_reward_gamma=2, baseline_min_len=2, length_punish_scale=0.1,
                       dropout_p=0.1, learning_rate=0.001,
                       epoches=20, saved_name='test_structed_rl_model.pkl', load_name=None,
                       gcc_file_path='test.c', normalize_type='layer',
                       predict_type='first', pretrain_s_model_epoch=2)




