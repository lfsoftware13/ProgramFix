import os

from c_parser.buffered_clex import BufferedCLex
from c_parser.pycparser.pycparser import CParser
from c_parser.pycparser.pycparser.c_lexer import CLexer


class ValueToken:
    def __init__(self, value, type, lineno, lexpos):
        self.value = value
        self.type = type
        self.lineno = lineno
        self.lexpos = lexpos

    def __str__(self):
        return 'ValToken(%s,%r,%d,%d)' % (self.type, self.value, self.lineno, self.lexpos)

    def __repr__(self):
        return str(self)


def init_pycparser(lexer=CLexer):
    c_parser = CParser()
    c_parser.build(lexer=lexer)
    return c_parser


def tokenize_by_clex_fn():
    c_parser = init_pycparser(lexer=BufferedCLex)
    def tokenize_fn(code):
        tokens = tokenize_by_clex(code, c_parser.clex)
        return tokens
    return tokenize_fn


tokenize_error_count = 0
count = 0
def tokenize_by_clex(code, lexer):
    # print('code: ', code)
    global tokenize_error_count, count
    try:
        if count % 1000 == 0:
            print('tokenize: {}'.format(count))
        count += 1
        lexer.reset_lineno()
        lexer.input(code)
        tokens = list(zip(*lexer._tokens_buffer))[0]
        return tokens
    except IndexError as e:
        # print('IndexError: ', e)
        tokenize_error_count += 1
        # print('token_buffer_len:{}'.format(len(lexer._tokens_buffer)))
        return None
    except Exception as a:
        # print('error: ', a)
        tokenize_error_count += 1
        return None


def transform_LexToken(token):
    token = ValueToken(token.value, token.type, token.lineno, token.lexpos)
    return token


def transform_LexToken_list(tokens):
    return [transform_LexToken(tok) for tok in tokens]


# ----------------------- a copy of write code to file method in util to solve loop reference -------------------- #

def write_code_to_file(code, file_path):
    file_path = os.path.abspath(file_path)
    ensure_file_path(file_path)
    f = open(file_path, 'w')
    f.write(code)
    f.flush()
    f.close()
    return file_path


def ensure_file_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


if __name__ == '__main__':
    # tokenize_fn = tokenize_by_clex_fn()
    # code = r'''int main ( int argc , char * * argv ) { int input [ 101 ] ; int tmp ; int n ; int i ; input [ 0 ] = 0 ; scanf ( "%d" , & n ) ; for ( i = 1 ; i <= n ; ++ i ) { scanf ( "%d" , & tmp ) ; input [ i ] = tmp + input [ i - 1 ] ; } if ( input [ n ] != 0 ) { printf ( "YES\n" ) ; printf ( "1\n1 %d\n" , n ) ; return 0 ; } for ( i = n - 1 ; i > 0 ; i -- ) { if ( input [ i + 1 ] == 0 && input [ i ] != 0 ) { printf ( "YES\n" ) ; printf ( "2\n1 %d\n%d %d\n" , i , i + 1 , n ) ; return 0 ; } } printf ( "NO\n" ) ; return 0 ; }'''
    # print('start')
    # for i in range(10):
    #     res = tokenize_fn(code)
    # print(transform_LexToken_list(res))
    with open(r'.\c_parser\fake_c_header\stdio.h') as f:
        test = f.read()
    c_parser = init_pycparser(lexer=BufferedCLex)
    res = c_parser.parse(test)
    print(c_parser._scope_stack)
