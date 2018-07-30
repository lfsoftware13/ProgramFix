import copy

from common.pycparser_util import tokenize_by_clex_fn, ValueToken
from read_data.load_data_vocabulary import create_common_error_vocabulary
from vocabulary.transform_vocabulary_and_parser import TransformVocabularyAndSLK


def main():
    begin_tokens = ['<BEGIN>']
    end_tokens = ['<END>']
    unk_token = '<UNK>'
    addition_tokens = ['<GAP>']
    vocabulary = create_common_error_vocabulary(begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                                                addition_tokens=addition_tokens)
    tokenize_fn = tokenize_by_clex_fn()
    transformer = TransformVocabularyAndSLK(vocabulary, tokenize_fn)

    code = r'''
    int main(){
        int a = 0;
        a = 1 
    '''
    token_list = tokenize_fn(code)
    print(token_list)
    token_list = iter(token_list)
    t_parser = transformer.create_new_slk_iterator()
    for t, type_id in t_parser:
        print(t)
        print(type_id)
        try:
            tt = next(token_list)
            print(tt)
            t_parser.add_token(tt)
        except StopIteration:
            break
    t_parser_1 = copy.deepcopy(t_parser)
    t_parser_2 = copy.deepcopy(t_parser)

    t_parser_1.add_token(transformer.id_to_token_dict[transformer.vocab.word_to_id('+')])
    t, type_id = next(t_parser_1)
    print('t1', t)
    print('t1', type_id)

    t_parser_2.add_token(transformer.id_to_token_dict[transformer.vocab.word_to_id(';')])
    t, type_id = next(t_parser_2)
    print('t2', t)
    print('t2', type_id)

if __name__ == '__main__':
    # main()

    def a():
        l = [1, 2, 3, 4, 5]
        for i in l:
            print('in a: {}'.format(i))
            yield i
            print('in a after: {}'.format(i))

    for i in a():
        print('in for : {}'.format(i))
