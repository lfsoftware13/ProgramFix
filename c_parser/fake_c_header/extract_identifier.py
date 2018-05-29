from c_parser.buffered_clex import BufferedCLex
from common.constants import ROOT_PATH
from common.pycparser_util import init_pycparser

import os


def extract_identifier(code):
    c_parser = init_pycparser(lexer=BufferedCLex)
    c_parser.parse(code)
    global_scope = c_parser._scope_stack[0]
    ids = list(zip(*list(filter(lambda x: not x[1], global_scope.items()))))[0]
    types = list(zip(*list(filter(lambda x: x[1], global_scope.items()))))[0]
    print(ids)
    print(types)
    return ids, types


def extract_fake_c_header_identifier():
    file_list = ['math.h', 'stdio.h', 'stdlib.h', 'string.h']
    res_ids = set()
    res_types = set()
    for fp in file_list:
        afp = os.path.join(ROOT_PATH, 'c_code_processer', 'fake_c_header', fp)
        with open(afp) as f:
            text = f.read()
            ids, types = extract_identifier(text)
            res_ids |= set(ids)
            res_types |= set(types)
    return res_ids, res_types


if __name__ == '__main__':
    res = extract_fake_c_header_identifier()
    print(res)
    print(len(res[0]), len(res[1]))
