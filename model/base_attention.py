import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.logger import info


def to_numpy(var):
    namelist = torch.typename(var).split('.')
    if "sparse" in namelist:
        var = var.to_dense()
    return var.cpu().numpy()

HAS_NAN = False
def is_nan(var):
    var = var.detach()
    if var is None:
        return "None"
    res = torch.isnan(torch.sum(var)).item()
    if res == 1:
        global HAS_NAN
        HAS_NAN = True
        res = True
    # res = np.isnan(np.sum(to_numpy(var)))
    # if res:
    #     global HAS_NAN
    #     HAS_NAN = True
    return res

def show_tensor(var):
    if var is None:
        return "None"
    var = to_numpy(var)
    return "all zeros:{}, has nan:{}, value:{}".format(np.all(var==0), np.isnan(np.sum(var)), var)

def create_hook_fn(name):
    def p(v):
        v = v.detach()
        nan_result = is_nan(v)
        info("{} gradient: is nan {}, max value: {}, min value: {}".format(name, nan_result, torch.max(v).item(), torch.min(v).item()))
        print("{} gradient: is nan {}, max value: {}, min value: {}".format(name, nan_result, torch.max(v).item(), torch.min(v).item()))
        # if nan_result:
        #     print(v)
        # print("{} gradient: is nan {}".format(name, is_nan(v.detach())))
        # print("{} gradient: is nan {}".format(name, v.detach()))
    return p


def register_hook(var, name):
    var.register_hook(create_hook_fn(name))


def record_is_nan(var, name):
    info('{}: {}'.format(name, str(is_nan(var))))
    register_hook(var, name + ' in hook: ')
    pass


class ScaledDotProductAttention(nn.Module):

    def __init__(self, value_hidden_size, output_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.output_dim = output_dim
        self.value_hidden_size = value_hidden_size
        self.transform_output = nn.Linear(value_hidden_size, output_dim)

    def forward(self, query, key, value, value_mask=None):
        """
        query = [batch, -1, query_seq, query_hidden_size]
        key = [batch, -1, value_seq, query_hidden_size]
        value = [batch, -1, value_seq, value_hidden_size]
        the len(shape) of query, key, value must be the same
        Create attention value of query_sequence according to value_sequence(memory).
        :param query:
        :param key:
        :param value:
        :param value_mask: [batch, max_len]. consist of 0 and 1.
        :return: attention_value = [batch, -1, query_seq, output_hidden_size]
        """
        query_shape = list(query.shape)
        key_shape = list(key.shape)
        value_shape = list(value.shape)
        scaled_value = math.sqrt(key_shape[-1])

        # [batch, -1, query_hidden_size, value_seq] <- [batch, -1, value_seq, query_hidden_size]
        key = torch.transpose(key, dim0=-2, dim1=-1)

        # [batch, -1, query_seq, value_seq] = [batch, -1, query_seq, query_hidden_size] * [batch, -1, query_hidden_size, value_seq]
        query_3d = query.contiguous().view(-1, query_shape[-2], query_shape[-1])
        key_3d = key.contiguous().view(-1, key_shape[-1], key_shape[-2])
        qk_dotproduct = torch.bmm(query_3d, key_3d).view(*query_shape[:-2], query_shape[-2], key_shape[-2])
        scaled_qk_dotproduct = qk_dotproduct/scaled_value
        # record_is_nan(scaled_qk_dotproduct, 'scaled_qk_dotproduct in ScaledDotProductAttention')

        # mask the padded token value
        if value_mask is not None:
            dim_len = len(list(scaled_qk_dotproduct.shape))
            # masked_fill = value_mask.view(value_shape[0], *[1 for i in range(dim_len-2)], value_shape[-2])
            # scaled_qk_dotproduct = torch.where(masked_fill, scaled_qk_dotproduct, to_cuda(torch.Tensor([float('-inf')])))
            scaled_qk_dotproduct.masked_fill_(~value_mask.view(value_mask.shape[0], *[1 for i in range(dim_len-2)], value_mask.shape[-1]), -float('inf'))
        # value_mask_sum = torch.sum(value_mask, dim=-1)
        # info(torch.eq(value_mask_sum, 0))
        # info('value_mask batch sum: {}, equal 0 sum: {}'.format(value_mask_sum, torch.sum(torch.eq(value_mask_sum, 0))))
        # record_is_nan(scaled_qk_dotproduct, 'scaled_qk_dotproduct after masked in ScaledDotProductAttention')


        weight_distribute = F.softmax(scaled_qk_dotproduct, dim=-1)
        # record_is_nan(weight_distribute, 'weight_distribute in ScaledDotProductAttention')
        weight_shape = list(weight_distribute.shape)
        attention_value = torch.bmm(weight_distribute.view(-1, *weight_shape[-2:]), value.contiguous().view(-1, *value_shape[-2:]))
        # record_is_nan(attention_value, 'attention_value in ScaledDotProductAttention')
        attention_value = attention_value.view(*weight_shape[:-2], *list(attention_value.shape)[-2:])
        # record_is_nan(attention_value, 'attention_value after view in ScaledDotProductAttention')
        transformed_output = self.transform_output(attention_value)
        # record_is_nan(transformed_output, 'transformed_output in ScaledDotProductAttention')
        return transformed_output


class OneHeaderAttention(nn.Module):
    def __init__(self, hidden_size, attention_type='scaled_dot_product'):
        super(OneHeaderAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        if attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention(self.hidden_size, self.hidden_size)
        else:
            raise Exception('no such attention_type: {}'.format(attention_type))

    def forward(self, inputs, memory, memory_mask=None):
        """
        query = [batch, sequence, hidden_size]
        memory = [batch, sequence, hidden_size]

        :param query:
        :param key:
        :param value:
        :return:
        """
        # info('memory_mask in OneHeaderAttention: , batch mask: {}'.format(torch.sum(memory_mask, dim=-1)))
        query = self.query_linear(inputs)
        key = self.key_linear(memory)
        value = self.value_linear(memory)
        # record_is_nan(query, 'query in OneHeaderAttention')
        # record_is_nan(key, 'key in OneHeaderAttention')
        # record_is_nan(value, 'value in OneHeaderAttention')
        atte_value = self.attention.forward(query, key, value, value_mask=memory_mask)
        # record_is_nan(atte_value, 'atte_value in OneHeaderAttention')
        return atte_value


class TrueMaskedMultiHeaderAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, attention_type='scaled_dot_product'):
        super(TrueMaskedMultiHeaderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.header_attention_list = nn.ModuleList([OneHeaderAttention(hidden_size, attention_type) for _ in range(num_heads)])
        self.output_linear = nn.Linear(num_heads * hidden_size, hidden_size)

    def forward(self, inputs, memory, memory_mask=None):
        """
        query = [batch, sequence, hidden_size]
        memory = [batch, sequence, hidden_size]

        :param query:
        :param key:
        :param value:
        :return:
        """
        # record_is_nan(inputs, 'inputs in TrueMaskedMultiHeaderAttention')
        # record_is_nan(memory, 'memory in TrueMaskedMultiHeaderAttention')
        # info('memory_mask in TrueMaskedMultiHeaderAttention: , batch mask: {}'.format(torch.sum(memory_mask, dim=-1)))
        atte_value_list = [m(inputs, memory, memory_mask) for m in self.header_attention_list]
        # for i, atte_value in enumerate(atte_value_list):
        #     record_is_nan(atte_value, 'atte_value in TrueMaskedMultiHeaderAttention' + str(i))

        atte_value = torch.cat(atte_value_list, dim=-1)
        # record_is_nan(atte_value, 'atte_value after concat in TrueMaskedMultiHeaderAttention')

        output_value = self.output_linear(atte_value)
        # record_is_nan(output_value, 'output_value in TrueMaskedMultiHeaderAttention')
        return output_value


class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, hidden_layer_count=1):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param hidden_layer_count: must >= 1
        """
        super(PositionWiseFeedForwardNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        if hidden_layer_count <= 0:
            raise Exception('at least one hidden layer')
        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        for i in range(hidden_layer_count-1):
            self.ff.add_module('hidden_' + str(i), nn.Linear(hidden_size, hidden_size))
            self.ff.add_module('relu_' + str(i), nn.ReLU())

        self.ff.add_module('output', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.ff(x)


class SelfAttentionEncoder(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(SelfAttentionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)

        self.relu = nn.ReLU()
        self.self_attention = TrueMaskedMultiHeaderAttention(hidden_size, num_heads)
        self.ff = PositionWiseFeedForwardNet(hidden_size, hidden_size, hidden_size, hidden_layer_count=1)
        if normalize_type == 'layer':
            self.self_attention_normalize = nn.LayerNorm(normalized_shape=hidden_size)
            self.ff_normalize = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, input, input_mask):
        # record_is_nan(input, 'input in SelfAttentionEncoder')
        atte_value = self.dropout(self.self_attention.forward(input, input, memory_mask=input_mask))
        # record_is_nan(atte_value, 'atte_value in SelfAttentionEncoder')
        if self.normalize_type is not None:
            atte_value = self.self_attention_normalize(atte_value) + atte_value
            atte_value = self.relu(atte_value)
            # record_is_nan(atte_value, 'atte_value after normalize in SelfAttentionEncoder')

        ff_value = self.dropout(self.ff.forward(atte_value))
        info('ff_value max value: {}, min value: {}'.format(torch.max(ff_value), torch.min(ff_value)))
        # record_is_nan(ff_value, 'ff_value in SelfAttentionEncoder')
        if self.normalize_type is not None:
            ff_value = self.ff_normalize(ff_value) + ff_value
            ff_value = self.relu(ff_value)
            # info('ff_value after normalize max value: {}, min value: {}'.format(torch.max(ff_value), torch.min(ff_value)))
            # record_is_nan(ff_value, 'ff_value after normalize in SelfAttentionEncoder')
        return ff_value


class TransformEncoderModel(nn.Module):
    def __init__(self, hidden_size, encoder_stack_num, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(TransformEncoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_stack_num = encoder_stack_num
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.normalize_type = normalize_type

        self.encoder_list = nn.ModuleList([SelfAttentionEncoder(hidden_size=self.hidden_size, num_heads=self.num_heads, dropout_p=self.dropout_p, normalize_type=normalize_type) for _ in range(self.encoder_stack_num)])

    def forward(self, embedded_input, input_mask):
        # encoder for n times
        # info('input_mask in TransformEncoderModel: , batch mask: {}'.format(torch.sum(input_mask, dim=-1)))
        encode_value = embedded_input
        # record_is_nan(encode_value, 'encode_value_0 in TransformEncoderModel')
        # count = 1
        for encoder in self.encoder_list:
            encode_value = encoder(encode_value, input_mask)
            # record_is_nan(encode_value, 'encode_value_'+str(count)+' in TransformEncoderModel')
            # count += 1
        return encode_value


class SelfAttentionDecoder(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(SelfAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)

        self.relu = nn.ReLU()
        self.input_self_attention = TrueMaskedMultiHeaderAttention(hidden_size, num_heads)
        self.attention = TrueMaskedMultiHeaderAttention(hidden_size, num_heads)
        self.ff = PositionWiseFeedForwardNet(hidden_size, hidden_size, hidden_size, hidden_layer_count=1)

        if normalize_type is not None:
            self.self_attention_normalize = nn.LayerNorm(normalized_shape=hidden_size)
            self.attention_normalize = nn.LayerNorm(normalized_shape=hidden_size)
            self.ff_normalize = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, input, input_mask, encoder_output, encoder_mask):
        self_atte_value = self.dropout(self.input_self_attention(input, input, input_mask))
        if self.normalize_type is not None:
            self_atte_value = self.self_attention_normalize(self_atte_value) + self_atte_value
            self_atte_value = self.relu(self_atte_value)

        atte_value = self.dropout(self.attention(self_atte_value, encoder_output, encoder_mask))
        if self.normalize_type is not None:
            atte_value = self.attention_normalize(atte_value) + atte_value
            # atte_value = self.relu(atte_value)

        ff_value = self.dropout(self.ff(atte_value))
        if self.normalize_type is not None:
            ff_value = self.ff_normalize(ff_value) + ff_value
            # ff_value = self.relu(ff_value)

        return ff_value


class TransformDecoderModel(nn.Module):
    def __init__(self, hidden_size, decoder_stack_num, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(TransformDecoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_stack_num = decoder_stack_num
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.normalize_type = normalize_type

        self.decoder_list = nn.ModuleList([SelfAttentionDecoder(hidden_size=self.hidden_size, num_heads=self.num_heads, dropout_p=self.dropout_p, normalize_type=normalize_type) for _ in range(self.decoder_stack_num)])

    def forward(self, position_output_embed, output_mask, encode_value, input_mask):
        # decoder for n times
        decoder_value = position_output_embed
        for decoder in self.decoder_list:
            decoder_value = decoder(decoder_value, output_mask, encode_value, input_mask)
        return decoder_value