from c_parser.slk_parser import PackedDynamicSLKParser
from common.constants import pre_defined_c_tokens, pre_defined_c_tokens_map, pre_defined_c_label, \
    pre_defined_c_library_tokens, CACHE_DATA_PATH
from common.logger import info
from common.pycparser_util import transform_LexToken_list, transform_LexToken
from common.util import create_token_mask_by_token_set, disk_cache, generate_mask


class TransformVocabularyAndSLK(object):

    def __init__(self, vocab, tokenize_fn):
        self.vocab = vocab

        self.tokenize_fn = tokenize_fn
        self.string_vocabulary_set = create_string_vocabulary_set(vocab, tokenize_fn)
        self.constant_vocabulary_set = create_constant_vocabulary_set(vocab, tokenize_fn)
        self.id_vocabulary_set = create_id_vocabulary_set(vocab, tokenize_fn)
        self.end_label_vocabulary_set = create_end_label_vocabulary_set(vocab, tokenize_fn)
        self.keyword_vocabulary_dict = create_keyword_vocabulary_dict(vocab)
        self.pre_defined_c_library_set = create_pre_defined_c_library_vocabulary_set(vocab)

        self.id_to_token_dict = create_ids_to_token_dict(vocab, tokenize_fn)

        self.parser = PackedDynamicSLKParser()
        self.slk_list = []

    def filter_code_ids(self, code_ids, code_length, start_pos):
        code_ids = code_ids.tolist()
        code_length = code_length.tolist()
        code_ids = [ids[start_pos:start_pos+l] for ids, l in zip(code_ids, code_length)]
        return code_ids

    def filter_code_ids_python(self, code_ids):
        code_ids = [ids[1:-1] for ids in code_ids]
        return code_ids

    def parse_ids_to_code(self, code_ids_list):
        code_values = [[self.vocab.id_to_word(i) for i in ids] for ids in code_ids_list]
        codes = [' '.join(v_list) for v_list in code_values]
        return codes

    def create_tokens_list(self, code_list):
        tokens_list = [self.tokenize_fn(code) for code in code_list]
        tokens_list = [transform_LexToken_list(tokens) for tokens in tokens_list]
        return tokens_list

    def create_id_constant_string_set_id_by_ids(self, code_ids):
        total_set = set(code_ids)
        id_set = total_set & self.id_vocabulary_set
        string_set = total_set & self.string_vocabulary_set
        constant_set = total_set & self.constant_vocabulary_set
        return id_set, string_set, constant_set

    def convert_slk_type_to_token_set(self, slk_type, id_set, string_set, constant_set):
        total_id_set = set()
        if 'END_OF_SLK_INPUT' in slk_type:
            total_id_set |= self.end_label_vocabulary_set
        if 'ID' in slk_type:
            total_id_set |= id_set
            total_id_set |= self.pre_defined_c_library_set
        if 'STRING_LITERAL' in slk_type:
            total_id_set |= string_set
        if 'CONSTANT' in slk_type:
            total_id_set |= constant_set
        keyword_set = set()
        for t in slk_type:
            if t in self.keyword_vocabulary_dict.keys():
                keyword_set |= {self.keyword_vocabulary_dict[t]}
        total_id_set |= keyword_set
        return total_id_set

    def get_slk_result(self, tokens):
        slk_res = []
        try:
            slk_res = self.parser.get_all_compatible_token(tokens)
        except Exception as e:
            slk_res = [[] for i in range(len(tokens)+1)]
            print(e)
            info(str(e))
            tokens_str = ' '.join([tok.value for tok in tokens])
            print(tokens_str)
            info(tokens_str)
        return slk_res

    def create_token_mask(self, s):
        if len(s) == 0:
            # print('token set mask is empty')
            # info('token set mask is empty')
            return generate_mask([(0, self.vocab.vocabulary_size-1)], size=self.vocab.vocabulary_size)
        # mask = create_token_mask_by_token_set(s, vocab_len=self.vocab.vocabulary_size)
        mask = generate_mask(s, size=self.vocab.vocabulary_size)
        return mask

    def get_all_token_mask_train(self, ac_tokens_ids, ac_length, start_pos=0):
        # token_ids_list = self.filter_code_ids(ac_tokens_ids, ac_length, start_pos=start_pos)
        token_ids_list = self.filter_code_ids_python(ac_tokens_ids)
        combine_result = [self.create_id_constant_string_set_id_by_ids(token_ids) for token_ids in token_ids_list]
        id_set_list, string_set_list, constant_set_list = list(zip(*combine_result))
        tokens_list = [[self.id_to_token_dict[i] for i in token_ids] for token_ids in token_ids_list]
        # codes_list = self.parse_ids_to_code(token_ids_list)
        # tokens_list = self.create_tokens_list(codes_list)
        slk_result_list = [self.get_slk_result(tokens) for tokens in tokens_list]
        token_id_set = [
            [self.convert_slk_type_to_token_set(res, id_set, string_set, constant_set) for res in slk_result]
            for slk_result, id_set, string_set, constant_set in
            zip(slk_result_list, id_set_list, string_set_list, constant_set_list)]

        token_id_mask = [[self.create_token_mask(s) for s in id_sets]
                         for id_sets in token_id_set]
        return token_id_mask


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


@disk_cache(basename='create_id_vocabulary_set', directory=CACHE_DATA_PATH)
def create_id_vocabulary_set(vocab, tokenize_fn):
    constant_label = ['ID', 'TYPEID']
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


@disk_cache(basename='create_pre_defined_c_library_vocabulary_set', directory=CACHE_DATA_PATH)
def create_pre_defined_c_library_vocabulary_set(vocab):
    library_set = set()
    for word in pre_defined_c_library_tokens:
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
    print('vocabulary: ', vocab.id_to_word(30600))
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






