import more_itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz.sandbox import unzip

from c_parser.ast_parser import parse_ast_code_graph
from common.problem_util import to_cuda
from common.torch_util import create_sequence_length_mask
from common.util import PaddedList
from model.pretrain_model.detect_error_model import ErrorDetectorModel
from model.pretrain_model.masked_language_model import MaskedLanguageModel


def set_model_requires_grad(m, requires_grad):
    for p in m.parameters():
        p.requires_grad = requires_grad
    if requires_grad:
        m.train()
    else:
        m.eval()


class PretrainMaskedCodeModel(nn.Module):
    def __init__(self, vocabulary, mask_language_model_param, detect_token_model_param, train_type, ignore_id, pad_id,
                 check_error_task):
        '''
        :param vocabulary:
        :param mask_language_model_param:
        :param detect_token_model_param:
        :param train_type: 'gene', 'only_disc', 'both', 'none'
        '''
        super().__init__()
        self.vocabulary = vocabulary
        self.generator = MaskedLanguageModel(**mask_language_model_param)
        self.discriminator = ErrorDetectorModel(**detect_token_model_param)
        self.ignore_id = ignore_id
        self.pad_id = pad_id
        self.check_error_task = check_error_task
        from common.pycparser_util import tokenize_by_clex_fn
        self.tokenize_fn = tokenize_by_clex_fn()

        self.train_type = ''
        self.change_model_train_type(train_type)

    def change_model_train_type(self, train_type):
        if self.train_type == train_type:
            return
        self.train_type = train_type

        if self.train_type == 'gene':
            set_model_requires_grad(self.generator, True)
            set_model_requires_grad(self.discriminator, False)
        elif self.train_type == 'only_disc' or self.train_type == 'disc' or self.train_type == 'bert':
            set_model_requires_grad(self.generator, False)
            set_model_requires_grad(self.discriminator, True)
        elif self.train_type == 'both':
            set_model_requires_grad(self.generator, True)
            set_model_requires_grad(self.discriminator, True)
        elif self.train_type == 'none':
            set_model_requires_grad(self.generator, False)
            set_model_requires_grad(self.discriminator, False)

    def forward(self, inp_seq, inp_seq_len, inp_mask, target_seq, target_len, target_mask, masked_positions,
                ac_seq, ac_seq_len):
        if self.train_type == 'both' or self.train_type == 'gene' or self.train_type == 'disc':
            masked_output_logit = self.generator(inp_seq, inp_seq_len, inp_mask, target_seq, target_len, target_mask, masked_positions)

            masked_output_ids = self.extract_result_from_logit(masked_output_logit[0])
            output_tokens_seq = self.create_gene_output_code_seq(inp_seq, target_seq, masked_output_ids)
        elif self.train_type == 'only_disc' or self.train_type == 'bert':
            masked_output_logit = [torch.ones([1]).to(inp_seq.device)]
            output_tokens_seq = inp_seq

        if self.train_type == 'both' or self.train_type == 'disc' or self.train_type == 'only_disc' or self.train_type == 'bert':
            disc_inputs, disc_is_effect = self.prepare_discriminator_input(output_tokens_seq, inp_seq_len)
            if self.train_type == 'bert':
                disc_targets = self.prepare_discriminator_target_bert(target_seq, disc_is_effect, self.ignore_id)
            else:
                disc_targets = self.prepare_discriminator_target(disc_inputs[1], ac_seq, ac_seq_len, self.check_error_task,
                                                                 disc_is_effect)

            disc_outputs_logits = self.discriminator(*disc_inputs)
        else:
            disc_outputs_logits = [torch.ones([1]).to(inp_seq.device)]
            disc_inputs = [torch.ones([1]).to(inp_seq.device)]
            disc_targets = [torch.ones([1]).to(inp_seq.device)]
        return masked_output_logit, disc_outputs_logits, disc_inputs, disc_targets

    def prepare_discriminator_target_bert(self, target_seq, is_effect, ignore_id):
        is_effect_mask = torch.ByteTensor(is_effect).unsqueeze(-1).to(target_seq.device)
        mask = target_seq.ne(0) * is_effect_mask
        target = torch.where(mask, target_seq, torch.LongTensor([ignore_id]).to(target_seq.device))
        return [target]

    def prepare_discriminator_input(self, output_tokens_seq, inp_seq_len):
        output_length_list = inp_seq_len.tolist()
        output_seq_list = self.transform_input_tensor_to_list(output_tokens_seq, output_length_list)

        output_names_list = [[self.vocabulary.id_to_word(i) for i in one] for one in output_seq_list]

        res = [generate_one_graph_input(one, self.vocabulary, self.tokenize_fn) for one in output_names_list]
        disc_input_names, disc_input_length, disc_adj, disc_is_effect = list(zip(*res))

        if self.train_type == 'only_disc' or self.train_type == 'bert':
            disc_input_names = [[i if i != 'mask' else '<MASK>' for i in one] for one in disc_input_names]

        disc_input_seq = [[self.vocabulary.word_to_id(i) for i in one] for one in disc_input_names]

        disc_inputs = parse_graph_input_from_mask_lm_output(disc_input_seq, disc_input_length, disc_adj)

        return disc_inputs, disc_is_effect

    def prepare_discriminator_target(self, output_tokens_seq, ac_seq, ac_seq_len, check_error_task, is_effect):
        gene_target_seq = parse_graph_output_from_mask_lm_output(output_tokens_seq, ac_seq, ac_seq_len,
                                                                 ignore_id=self.ignore_id,
                                                                 check_error_task=check_error_task, is_effect=is_effect)
        return [gene_target_seq]

    def extract_result_from_logit(self, seq_logit):
        output_seq = torch.squeeze(torch.topk(F.softmax(seq_logit, dim=-1), dim=-1, k=1)[1], dim=-1)
        return output_seq

    def create_gene_output_code_seq(self, input_seq, target_seq, gene_output_seq):
        masked_positions = torch.ne(target_seq, self.ignore_id)
        output_tokens_seq = torch.where(masked_positions, gene_output_seq, input_seq)
        return output_tokens_seq

    def transform_input_tensor_to_list(self, input_seq, input_length):
        input_seq_list = input_seq.tolist()
        input_seq_list = [inp[:l] for inp, l in zip(input_seq_list, input_length)]
        return input_seq_list


def generate_one_graph_input(input_token_names, vocabulary, tokenize_fn):
    begin_token = vocabulary.begin_tokens[0]
    end_token = vocabulary.end_tokens[0]
    if tokenize_fn(' '.join(input_token_names), print_info=False) is None:
        input_seq = [begin_token] + input_token_names + ['<Delimiter>', end_token]
        input_length = len(input_seq)
        adj = [[0,1]]
        is_effect = False
    else:
        code_graph, is_effect = parse_ast_code_graph(input_token_names)
        input_length = code_graph.graph_length + 2
        in_seq, graph = code_graph.graph
        input_seq = [begin_token] + in_seq + [end_token]
        adj = [[a + 1, b + 1] for a, b, _ in graph] + [[b + 1, a + 1] for a, b, _ in graph]
    return input_seq, input_length, adj, is_effect


def parse_graph_input_from_mask_lm_output(input_seq, input_length, adj, use_ast=True):
    from common.problem_util import to_cuda
    from common.util import PaddedList

    def to_long(x):
        return to_cuda(torch.LongTensor(x))

    if not use_ast:
        adjacent_matrix = to_long(adj)
    else:
        adjacent_tuple = [[[i] + tt for tt in t] for i, t in enumerate(adj)]
        adjacent_tuple = [list(t) for t in unzip(more_itertools.flatten(adjacent_tuple))]
        size = max(input_length)
        # print("max length in this batch:{}".format(size))
        adjacent_tuple = torch.LongTensor(adjacent_tuple)
        adjacent_values = torch.ones(adjacent_tuple.shape[1]).long()
        adjacent_size = torch.Size([len(input_length), size, size])
        # info('batch_data input_length: ' + str(batch_data['input_length']))
        # info('size: ' + str(size))
        # info('adjacent_tuple: ' + str(adjacent_tuple.shape))
        # info('adjacent_size: ' + str(adjacent_size))
        adjacent_matrix = to_cuda(
            torch.sparse.LongTensor(
                adjacent_tuple,
                adjacent_values,
                adjacent_size,
            ).float().to_dense()
        )
    input_seq = to_long(PaddedList(input_seq))
    input_length = to_long(input_length)
    return adjacent_matrix, input_seq, input_length


def parse_graph_output_from_mask_lm_output(input_seq, ac_seq, ac_seq_len, ignore_id=-1, check_error_task=False, is_effect=[]):
    is_effect_mask = torch.ByteTensor(is_effect).unsqueeze(-1).to(ac_seq_len.device)
    mask = create_sequence_length_mask(ac_seq_len)
    mask = mask * is_effect_mask
    if check_error_task:
        target = torch.eq(input_seq[:, 1:1+ac_seq.size(1)], ac_seq).float()
        target = torch.where(mask, target, torch.FloatTensor([ignore_id]).to(target.device))
    else:
        target = torch.where(mask, ac_seq, torch.LongTensor([ignore_id]).to(ac_seq.device))
    return target


def create_parse_input_batch_data_fn():
    def parse_input_tensor(batch_data, do_sample=False):
        input_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['input_seq'])))
        inp_seq_len = to_cuda(torch.LongTensor(batch_data['input_seq_len']))
        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_seq'])))
        target_seq_len = to_cuda(torch.LongTensor(batch_data['target_seq_len']))
        ac_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_seq'])))
        ac_seq_len = to_cuda(torch.LongTensor(PaddedList(batch_data['ac_seq_len'])))
        return input_seq, inp_seq_len, None, target_seq, target_seq_len, None, batch_data['masked_positions'], \
               ac_seq, ac_seq_len
    return parse_input_tensor


def create_parse_target_batch_data_fn(ignore_id):
    def parse_target_tensor(batch_data):
        masked_target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_seq'], fill_value=ignore_id)))
        return [masked_target_seq]
    return parse_target_tensor


def create_loss_fn(ignore_id, check_error_task=True, train_type='both'):
    seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
    check_error_loss = nn.BCEWithLogitsLoss(reduction='none')
    def loss_fn(masked_output_logit, disc_outputs_logits, disc_inputs, disc_targets, masked_target_seq):
        masked_output_logit = masked_output_logit[0]
        if train_type == 'both' or train_type == 'gene':
            masked_loss = seq_loss(masked_output_logit.view(-1, masked_output_logit.size(2)), masked_target_seq.view(-1))
        else:
            masked_loss = torch.FloatTensor([0]).to(masked_output_logit.device)

        if train_type == 'both' or train_type == 'disc' or train_type == 'only_disc' or train_type == 'bert':
            if check_error_task:
                position_target = disc_targets[0]
                disc_position_logits = disc_outputs_logits[0]
                disc_position_logits = disc_position_logits[:, 1:1+position_target.size(1)]
                mask = torch.ne(position_target, ignore_id)
                disc_loss = check_error_loss(disc_position_logits, position_target.float()) * mask.float()
                replaced_loss = 3 * (torch.sum(disc_loss) / torch.sum(mask.float()))
            else:
                seq_target = disc_targets[0]
                disc_seq_logits = disc_outputs_logits[0]
                disc_seq_logits = disc_seq_logits[:, 1:1 + seq_target.size(1)]
                replaced_loss = seq_loss(disc_seq_logits.contiguous().view(-1, disc_seq_logits.size(2)), seq_target.view(-1))
        else:
            replaced_loss = 3 * torch.FloatTensor([0]).to(masked_output_logit.device)

        combine_loss = masked_loss + replaced_loss
        return combine_loss
    return loss_fn


def create_output_ids_fn(train_type='both'):
    def create_output(model_output, model_input, do_sample):
        if train_type == 'only_disc' or train_type == 'bert':
            return model_input[0]

        input_seq = model_input[0]
        output_seq = model_output[0][0]
        output_seq = torch.squeeze(torch.topk(F.softmax(output_seq, dim=-1), dim=-1, k=1)[1], dim=-1)

        masked_positions = model_input[6]
        positions = [[(i, p) for p in positions] for i, positions in enumerate(masked_positions)]
        positions = torch.LongTensor(list(more_itertools.flatten(positions))).to(input_seq.device)

        mask = torch.ones(input_seq.size(), device=input_seq.device).byte()
        mask[positions[:,0],positions[:,1]] = 0

        result = torch.where(mask, input_seq, output_seq)
        return result
    return create_output
