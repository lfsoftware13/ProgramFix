import copy

import torch

from c_parser.slk_parser import PackedDynamicSLKParser
from common.constants import pre_defined_c_tokens, pre_defined_c_tokens_map, pre_defined_c_label, \
    pre_defined_c_library_tokens, CACHE_DATA_PATH, c_standard_library_defined_identifier, \
    c_standard_library_defined_types
from common.logger import info
from common.pycparser_util import transform_LexToken_list, transform_LexToken, tokenize_by_clex_fn
from common.util import create_token_mask_by_token_set, disk_cache, generate_mask, OrderedList
from read_data.load_data_vocabulary import create_common_error_vocabulary


class TransformVocabularyAndSLK(object):

    def __init__(self, vocab, tokenize_fn):
        self.vocab = vocab
        self.string_vocabulary_set = create_string_vocabulary_set(vocab, tokenize_fn)
        self.constant_vocabulary_set = create_constant_vocabulary_set(vocab, tokenize_fn)
        self.id_vocabulary_set = create_identifier_vocabulary_set(vocab, tokenize_fn)
        self.end_label_vocabulary_set = create_end_label_vocabulary_set(vocab, tokenize_fn)
        self.keyword_vocabulary_dict = create_keyword_vocabulary_dict(vocab)
        self.pre_defined_c_identifier_library_set = create_pre_defined_c_library_identifier_vocabulary_set(vocab)
        self.pre_defined_c_typeid_library_set = create_pre_defined_c_library_typeid_vocabulary_set(vocab)

        self.id_to_token_dict = create_ids_to_token_dict(vocab, tokenize_fn)

        self.parser = PackedDynamicSLKParser()
        self.slk_list = []

    def filter_code_ids(self, code_ids, code_length, start_pos):
        code_ids = code_ids.tolist()
        code_length = code_length.tolist()
        code_ids = [ids[start_pos:start_pos+l] for ids, l in zip(code_ids, code_length)]
        return code_ids

    def filter_code_ids_python(self, code_ids):
        code_ids = [ids[:] for ids in code_ids]
        return code_ids

    def parse_ids_to_code(self, code_ids_list):
        code_values = [[self.vocab.id_to_word(i) for i in ids] for ids in code_ids_list]
        codes = [' '.join(v_list) for v_list in code_values]
        return codes

    # def create_tokens_list(self, code_list):
    #     tokens_list = [self.tokenize_fn(code) for code in code_list]
    #     tokens_list = [transform_LexToken_list(tokens) for tokens in tokens_list]
    #     return tokens_list

    def create_id_constant_string_set_id_by_ids(self, code_ids):
        total_set = set(code_ids)
        id_set = total_set & self.id_vocabulary_set
        string_set = total_set & self.string_vocabulary_set
        constant_set = total_set & self.constant_vocabulary_set
        return id_set, string_set, constant_set

    def convert_slk_type_to_token_set(self, slk_type, id_set, string_set, constant_set, typeid_set):
        # total_id_set = {self.vocab.word_to_id(self.vocab.unk)}
        total_id_set = set()
        if 'END_OF_SLK_INPUT' in slk_type:
            total_id_set |= self.end_label_vocabulary_set
        if 'ID' in slk_type:
            tmp_id_set = id_set - typeid_set - self.pre_defined_c_typeid_library_set
            total_id_set |= tmp_id_set
            total_id_set |= self.pre_defined_c_identifier_library_set
        if 'TYPEID' in slk_type:
            total_id_set |= typeid_set
            total_id_set |= self.pre_defined_c_typeid_library_set
        if 'STRING_LITERAL' in slk_type:
            total_id_set |= string_set
        if 'CONSTANT' in slk_type:
            total_id_set |= constant_set
        keyword_set = set()
        for t in slk_type:
            if t in self.keyword_vocabulary_dict.keys():
                keyword_set |= {self.keyword_vocabulary_dict[t]}
        total_id_set |= keyword_set
        if len(total_id_set) == 1:
            pass
            # keyword_set = {v for v in self.keyword_vocabulary_dict.values()}
            # total_id_set = id_set | self.pre_defined_c_library_set | string_set | constant_set | keyword_set
            # print('token set mask is empty')
            # info('token set mask is empty')
        return OrderedList(total_id_set)

    def get_slk_result(self, tokens):
        slk_res = []
        try:
            slk_res, typeid_res = self.parser.get_all_compatible_token(tokens)
            typeid_set = [{self.vocab.word_to_id(x) for x in
                           filter(lambda x: x in self.vocab.word_to_id_dict.keys(), typeids)} for typeids in typeid_res]
        except Exception as e:
            # slk_res = [[] for i in range(len(tokens)+1)]
            # typeid_res = [[] for i in range(len(tokens)+1)]
            # print(e)
            info(str(e))
            tokens_str = ' '.join([tok.value for tok in tokens])
            # print(tokens_str)
            info(tokens_str)
            info([tok.type for tok in tokens])
            # print(tokens_str)
            # print([tok.type for tok in tokens])
            raise Exception('slk error: '+ str(e))
        return slk_res, typeid_set

    def create_token_mask(self, s):
        if len(s) == 0:
            # print('token set mask is empty')
            # info('token set mask is empty')
            return generate_mask([(0, self.vocab.vocabulary_size-1)], size=self.vocab.vocabulary_size)
        # mask = create_token_mask_by_token_set(s, vocab_len=self.vocab.vocabulary_size)
        mask = generate_mask(s, size=self.vocab.vocabulary_size)
        return mask

    def get_all_token_mask_train(self, ac_tokens_ids, ac_length=None, start_pos=0):
        if isinstance(ac_tokens_ids, torch.Tensor):
            token_ids_list = self.filter_code_ids(ac_tokens_ids, ac_length, start_pos=start_pos)
        else:
            # token_ids_list = self.filter_code_ids_python(ac_tokens_ids)
            token_ids_list = ac_tokens_ids
        combine_result = [self.create_id_constant_string_set_id_by_ids(token_ids) for token_ids in token_ids_list]
        id_set_list, string_set_list, constant_set_list = list(zip(*combine_result))
        tokens_list = [[copy.copy(self.id_to_token_dict[i]) for i in token_ids] for token_ids in token_ids_list]
        # codes_list = self.parse_ids_to_code(token_ids_list)
        # tokens_list = self.create_tokens_list(codes_list)
        slk_result_list = [self.get_slk_result(tokens) for tokens in tokens_list]
        token_true_id_res = [
            [self.convert_slk_type_to_token_set(res, id_set, string_set, constant_set, typeid_set=typeid_set) for res, typeid_set in zip(*slk_result)]
            for slk_result, id_set, string_set, constant_set in
            zip(slk_result_list, id_set_list, string_set_list, constant_set_list)]

        # token_true_id_list = [[sorted(one_ids) for one_ids in token_ids] for token_ids in token_true_id_set]

        return token_true_id_res

    def create_new_slk_iterator(self):
        return self.parser.new()

    def get_candicate_step(self, t_parser, token_id, previous_id_set_list):

        # id_set, string_set, constant_set = self.create_id_constant_string_set_id_by_ids(previous_token_ids)

        if token_id is not None:
            token = copy.copy(self.id_to_token_dict[token_id])
            # print(token)
            t_parser.add_token(token)
        slk_result, typeid_res = next(t_parser)
        typeid_set = {self.vocab.word_to_id(x) for x in
                      filter(lambda x: x in self.vocab.word_to_id_dict.keys(), typeid_res)}
        # print(slk_result)
        token_id_list = self.convert_slk_type_to_token_set(slk_result, *previous_id_set_list, typeid_set=typeid_set)
        return token_id_list


@disk_cache(basename='create_string_vocabulary_set', directory=CACHE_DATA_PATH)
def create_string_vocabulary_set(vocab, tokenize_fn):
    string_label = ['STRING_LITERAL', 'WSTRING_LITERAL']
    token_set = create_special_type_vocabulary_mask(vocab, tokenize_fn, string_label)
    return token_set


@disk_cache(basename='create_constant_vocabulary_set', directory=CACHE_DATA_PATH)
def create_constant_vocabulary_set(vocab, tokenize_fn):
    constant_label = ['INT_CONST_DEC', 'INT_CONST_OCT', 'INT_CONST_HEX', 'INT_CONST_BIN',
                      'FLOAT_CONST', 'HEX_FLOAT_CONST', 'CHAR_CONST', 'WCHAR_CONST']
    token_set = create_special_type_vocabulary_mask(vocab, tokenize_fn, constant_label)
    return token_set


@disk_cache(basename='create_identifier_vocabulary_set', directory=CACHE_DATA_PATH)
def create_identifier_vocabulary_set(vocab, tokenize_fn):
    constant_label = ['ID']
    token_set = create_special_type_vocabulary_mask(vocab, tokenize_fn, constant_label)
    return token_set


@disk_cache(basename='create_end_label_vocabulary_set', directory=CACHE_DATA_PATH)
def create_end_label_vocabulary_set(vocab, tokenize_fn):
    end_id = vocab.word_to_id(vocab.end_tokens[0])
    token_set = {end_id}
    return token_set


@disk_cache(basename='create_keyword_vocabulary_dict', directory=CACHE_DATA_PATH)
def create_keyword_vocabulary_dict(vocab):
    keyword_dict = {}
    for label, word in pre_defined_c_tokens_map.items():
        if word in vocab.word_to_id_dict.keys():
            keyword_dict[label] = vocab.word_to_id(word)
    return keyword_dict


@disk_cache(basename='create_pre_defined_c_library_identifier_vocabulary_set', directory=CACHE_DATA_PATH)
def create_pre_defined_c_library_identifier_vocabulary_set(vocab):
    library_set = set()
    for word in c_standard_library_defined_identifier:
        if word in vocab.word_to_id_dict.keys():
            library_set |= {vocab.word_to_id(word)}
    return library_set


@disk_cache(basename='create_pre_defined_c_library_typeid_vocabulary_set', directory=CACHE_DATA_PATH)
def create_pre_defined_c_library_typeid_vocabulary_set(vocab):
    library_set = set()
    for word in c_standard_library_defined_types:
        if word in vocab.word_to_id_dict.keys():
            library_set |= {vocab.word_to_id(word)}
    return library_set


@disk_cache(basename='create_ids_to_token_dict', directory=CACHE_DATA_PATH)
def create_ids_to_token_dict(vocab, tokenize_fn):
    special_token = set(vocab.begin_tokens) | set(vocab.end_tokens) | set(vocab.addition_tokens) | {vocab.unk}
    id_to_token_dict = {}
    for word, i in vocab.word_to_id_dict.items():
        if word not in special_token:
            tokens = tokenize_fn(word)
            if tokens is not None:
                tok = tokenize_fn(word)[0]
                tok = transform_LexToken(tok)
                id_to_token_dict[i] = tok
            else:
                continue
            if len(tokens) > 1:
                info('vocabulary tokenize length is {}'.format(len(tokens)))
    tok = tokenize_fn('unk')[0]
    id_to_token_dict[vocab.word_to_id(vocab.unk)] = transform_LexToken(tok)
    return id_to_token_dict



def create_special_type_vocabulary_mask(vocab, tokenize_fn, labels):
    special_token = set(vocab.begin_tokens) | set(vocab.end_tokens) | set(vocab.addition_tokens) | {vocab.unk}
    ids = []
    for word, i in vocab.word_to_id_dict.items():
        if word not in special_token:
            tokens = tokenize_fn(word)
            if tokens is not None:
                res = tokenize_fn(word)[0]
            else:
                continue
            if len(tokens) > 1:
                info('vocabulary tokenize length is {}'.format(len(tokens)))
            if res.type in labels:
                ids += [i]
    token_set = set(ids)
    return token_set


if __name__ == '__main__':
    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                                                addition_tokens=addition_tokens)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(vocabulary, tokenize_fn)


    code = r'''
         int main ( ) { char x [ 99 ] , y [ 99 ] , z [ 99 ] ; int i = 0 , l1 , l2 ; int k , l , u , B ; scanf ( "%s %s" , x , y ) ; l1 = strlen ( x ) ; l2 = strlen ( y ) ; if ( l1 == l2 ) { for ( i = 0 ; i < l1 ; i ++ ) { if ( x [ i ] >= 97 && x [ i ] <= 122 && y [ i ] >= 97 && y [ i ] <= 122 ) { if ( x [ i ] < y [ i ] ) { B = - 1 ; break ; } else if ( x [ i ] > y [ i ] ) { z [ i ] = y [ i ] ; } else if ( x [ i ] == y [ i ] ) { l = x [ i ] ; u = 122 ; srand ( ( unsigned ) time ( NULL ) ) ; z [ i ] = l + rand ( ) % ( u - l + 1 ) ; } } } } if ( B == - 1 ) { printf ( "%d" , B ) ; } else { printf ( "%s" , z ) ; } return 0 ; }
         '''
    name_list = [tok.value for tok in tokenize_fn(code)]
    ids_list = vocabulary.parse_text_without_pad([name_list])
    print(ids_list)
    transformer.get_all_token_mask_train(ids_list)

    id_to_token_dict = create_ids_to_token_dict(vocabulary, tokenize_fn)
    u_id = vocabulary.word_to_id(end_tokens[0])
    word = vocabulary.id_to_word(29947)
    print('end_id: {}'.format(u_id))
    # print('u id : {}, u token: {}'.format(u_id, id_to_token_dict[u_id]))

    print('word:{},token:{}'.format(word, id_to_token_dict[10394]))






