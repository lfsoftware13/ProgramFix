import abc
import re
import collections

import inspect
import types
import typing
from abc import abstractmethod, ABCMeta

from c_parser.pycparser import pycparser
import more_itertools
import cytoolz as toolz

from c_parser.buffered_clex import BufferedCLex
from c_parser.pycparser.pycparser import CParser
from common import util


def check_include_between_two_code(code1, code2):
    names1 = extract_include_from_code(code1)
    names2 = extract_include_from_code(code2)
    return equal_include(names1, names2)


def extract_include_from_code(code):
    includes = extract_include(code)
    include_names = [extract_include_name(inc) for inc in includes]
    return include_names


def remove_include(code):
    lines = code.split('\n')
    pattern = re.compile('#include *<(.*)>|#include *"(.*)"')
    lines_without_include = list(filter(lambda line: pattern.match(line.strip()) is None, lines))
    return '\n'.join(lines_without_include)


def replace_include_with_blank(code):
    lines = code.split('\n')
    pattern = re.compile('#include *<(.*)>|#include *"(.*)"')
    lines_without_include = [line if pattern.match(line.strip()) is None else '' for line in lines]
    return '\n'.join(lines_without_include)


def replace_define_with_blank(code):
    lines = code.split('\n')
    pattern = re.compile('#define .*')
    lines_without_define = [line if pattern.match(line.strip()) is None else '' for line in lines]
    return '\n'.join(lines_without_define)


def replace_hashtag_with_blank(code):
    lines = code.split('\n')
    lines_without_hashtag = [line if line.strip()[0] != '#' else '' for line in lines]
    return '\n'.join(lines_without_hashtag)


def analyse_include_line_no(code, include_lines):
    lines = code.split('\n')
    include_line_nos = [match_one_include_line_no(lines, include_line) for include_line in include_lines]
    return include_line_nos


def match_one_include_line_no(lines, include_line):
    for i in range(len(lines)):
        if lines[i] == include_line:
            return i
    print('match one include line no error. lines: {}, include_line:{}'.format(lines, include_line))
    return None


def equal_include(names1, names2):
    if len(names1) != len(names2):
        return False
    for inc1, inc2 in zip(names1, names2):
        if inc1 != inc2:
            return False
    return True


def extract_include(code):
    lines = code.split('\n')
    pattern = re.compile('#include *<(.*)>|#include *"(.*)"')
    lines = map(str.strip, lines)
    include_lines = list(filter(lambda line: pattern.match(line) is not None, lines))
    return include_lines


def extract_include_name(include):
    include = include.strip()
    m = re.match('#include *<(.*)>', include)
    if m:
        return m.group(1)
    m = re.match('#include *"(.*)"', include)
    if m:
        return m.group(1)
    return None


def tokenize(code: str):
    lex = BufferedCLex(lambda x1, x2, x3:None, lambda: None, lambda: None, lambda x:None)
    lex.build()
    lex.input(code)
    return lex.tokens_buffer


def create_is_p_fn():
    pattern = re.compile(r"p_.*")

    def is_p_fn(name):
        return pattern.match(name)
    return is_p_fn


class Production(object):
    def __init__(self, left: str, right: typing.List[str], token_id_map: typing.Dict[str, int]):
        self._left = left
        self._right = right
        self._left_id = token_id_map[left]
        self._right_id = [token_id_map[c] for c in right]

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def left_id(self):
        return self._left_id

    @property
    def right_id(self):
        return self._right_id

    def __str__(self):
        return "{}    : {}".format(self.left, " ".join(self.right))

    def __eq__(self, other):
        if not isinstance(other, Production):
            return False
        if self.left != other.left:
            return False
        if self.right != other.right:
            return False
        if self.left_id != other.left_id:
            return False
        if self.right_id != other.right_id:
            return False
        return True

    def __hash__(self):
        return self.left.__hash__() + self.left_id + sum(t.__hash__() for t in self.right) + sum(self.right_id)


class ProductionVocabulary(object, metaclass=ABCMeta):
    @abstractmethod
    def get_matched_terminal_node(self, token_id):
        pass

    @abstractmethod
    def token_num(self):
        pass

    @property
    @abstractmethod
    def EMPTY_id(self):
        pass


class C99ProductionVocabulary(ProductionVocabulary):
    @property
    def EMPTY_id(self):
        return self.get_token_id(self.EMPTY)

    def __init__(self,
                 production_list: typing.List,):
        self._token_set = set(i.strip() for i in more_itertools.collapse(production_list))
        self._id_token_map = self._get_set_id_map(self._token_set)
        self._EMPTY = "<EMPTY>" #This token is used to indicate the stack is empty
        self._token_set.add(self._EMPTY)
        self._id_token_map[len(self._id_token_map)] = self._EMPTY
        self._token_id_map = util.reverse_dict(self._id_token_map)
        self._production_list = [Production(left, right, self._token_id_map) for left, right in production_list]
        self._id_production_map = self._get_set_id_map(self._production_list)
        self._production_id_map = util.reverse_dict(self._id_production_map)
        self._token_derivate_map = toolz.groupby(lambda x: x.left_id, self._production_list)
        self._string_production_map = {str(production): production for production in self._production_list}
        self._terminal_set = set(i.strip() for i in pycparser.c_lexer.CLexer.tokens)
        self._terminal_set.add(self.EMPTY)
        self._terminal_id_set = set(self._token_id_map[t] for t in self._terminal_set)
        self._match_terminal_node = self._create_matched_ternimal_node()

    def _create_matched_ternimal_node(self,):
        token_dict = {}
        def _create_dict(token_id):
            if token_id in token_dict:
                return token_dict[token_id]
            token_dict[token_id] = set()
            if self.is_ternimal(token_id):
                token_dict[token_id] = {token_id}
                return token_dict[token_id]
            for p in self.get_matched_production(token_id):
                if len(p.right_id) >= 1:
                    token_dict[token_id] |= _create_dict(p.right_id[0])
            return token_dict[token_id]
        for token in self._token_set:
            _create_dict(self._token_id_map[token])
        for k, v in token_dict.items():
            for i_v in v:
                assert self.is_ternimal(i_v)
        return token_dict

    def _get_set_id_map(self, s):
        s = sorted(s, key=lambda x: str(x))
        return dict(enumerate(s))

    def get_production_by_id(self, i):
        return self._id_production_map[i]

    def get_matched_production(self, token_id):
        if token_id in self._token_derivate_map:
            return self._token_derivate_map[token_id]
        else:
            return []

    def get_matched_terminal_node(self, token_id):
        return self._match_terminal_node[token_id]

    def get_production_by_production_string(self, doc):
        """
        :param doc: a production string
        :return: a list of production in the production string
        """
        production_str_list = split_production_string(doc)
        productions = []
        for left, right in production_str_list:
            productions.append(self._string_production_map["{}    : {}".format(left, " ".join(right))])
        return productions

    def is_ternimal(self, i):
        return i in self._terminal_id_set

    def get_token_id(self, token):
        return self._token_id_map[token]

    def token_num(self):
        return len(self._token_set)

    def get_production_id(self, production):
        return self._production_id_map[production]

    def production_num(self):
        return len(self._production_list)

    @property
    def EMPTY(self):
        return self._EMPTY

    def __str__(self):
        return "\n".join([str(production) for production in self._production_list])


def split_production_string(s: str):
    left, rights = s.split(":")
    left = left.strip()
    rights = rights.split("|")
    rights = [re.split("\s+", right.strip()) for right in rights]
    lefts = [left] * len(rights)
    productions = list(zip(lefts, rights))
    productions = [(left, [] if right==[''] else right) for left, right in productions]
    return productions


def get_all_c99_production_vocabulary():
    is_p_fn = create_is_p_fn()
    parser = pycparser.CParser()
    parse_fn_tuple_list = filter(lambda x: is_p_fn(x[0]) and x[0] != "p_error", inspect.getmembers(parser))
    production_list = map(lambda x: x[1].__doc__, parse_fn_tuple_list)

    production_list = list(more_itertools.flatten(list(map(split_production_string, production_list))))

    production_vocabulary = C99ProductionVocabulary(production_list, )

    return production_vocabulary


def show_all_c99_productions():
    is_p_fn = create_is_p_fn()
    parser = pycparser.CParser()
    parse_fn_tuple_list = filter(lambda x: is_p_fn(x[0]) and x[0] != "p_error", inspect.getmembers(parser))
    production_list = map(lambda x: x[1].__doc__, parse_fn_tuple_list)
    # pattern = re.compile(":")
    for p in production_list:
        left, rights = p.split(":")
        left = left.strip()
        rights = rights.split("|")
        rights = [re.split("\s+", right.strip()) for right in rights]
        rights = [" ".join(right) for right in rights]
        rights = "|".join(rights)
        print("{} : {}".format(left, rights))


class ParseNode(metaclass=abc.ABCMeta):

    def __init__(self):
        self._parent_node = None

    @property
    def parent_node(self):
        """
        :return: The reference of the parent node
        """
        return self._parent_node

    @parent_node.setter
    def parent_node(self, n):
        """
        :param n: The parent node
        :return: None
        """
        self._parent_node = n

    @abc.abstractmethod
    def is_leaf(self):
        """
        :return: True if this is a leaf node, else False
        """

    @property
    @abc.abstractmethod
    def type_id(self):
        """
        :return: The id in the left of the production or the token type id in the leaf node
        """

    @property
    @abc.abstractmethod
    def type_string(self):
        """
        :return: The string in the left of the production or the token type string in the leaf node
        """


class ProductionNode(ParseNode):
    def __init__(self, production: typing.Union[Production, None], ast_node):
        """
        :param production: if production is None, this attribute should be filled latter
        :param ast_node:
        """
        super().__init__()
        self._ast_node = ast_node
        if production is not None:
            self.production = production
        self._children_nodes = {}

    def is_leaf(self):
        return False

    @property
    def production(self):
        return self._production

    @production.setter
    def production(self, production_):
        self._production = production_
        self._right_map = dict(zip(production_.right, production_.right_id))

    @property
    def type_id(self):
        return self.production.left_id

    @property
    def type_string(self):
        return self._production.left

    @property
    def children(self):
        return [self._children_nodes[i] for i in range(len(self.production.right_id))]

    def is_all_filled(self):
        for right_id in self.production.right_id:
            if right_id not in self._children_nodes:
                return False
        return True

    def _check_item(self, item):
        if isinstance(item, int):
            pass
        else:
            raise TypeError("The type of item in ParseNode should be int or str")
        if not len(self.production.right_id) > item >= 0:
            raise KeyError("The item is not in the production right")
        return item

    def __getitem__(self, item):
        item = self._check_item(item)
        return self._children_nodes[item]

    def __setitem__(self, key, value):
        item = self._check_item(key)
        self._children_nodes[item] = value

    def __repr__(self):
        return "{}".format(str(self.production))


class LeafParseNode(ParseNode):
    @property
    def type_id(self):
        return self._token_id

    @property
    def type_string(self):
        return self._token

    def __init__(self, type_string, value, type_id):
        super().__init__()
        self._token = type_string
        self._token_id = type_id
        self._value = value

    def is_leaf(self):
        return True

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def __repr__(self):
        return "{}:{}".format(self.type_string, self.value)


Token = collections.namedtuple("Token", ["type", "value"])


def show_production_node(node):
    prefix_tab = " "
    stack = [("", node)]
    while stack:
        tab, node = stack.pop()
        next_tab = tab + prefix_tab
        print(tab+node.type_string)
        if node.is_leaf():
            print(tab + node.value)
        else:
            for child in reversed(node.children):
                stack.append((next_tab, child))


LeafToken = collections.namedtuple("LeafToken", ["type_id", "type_string", "value"])


def parse_tree_to_top_down_process(node):
    """
    :param node: The root node of a parse tree
    :return: a list of node in this tree which is in preorder traversal
    """
    prefix_tab = " "
    stack = [("", node)]
    production_list = []
    # print(stack)
    while len(stack) != 0:
        # print(stack)
        tab, next_node = stack.pop()
        next_tab = tab + prefix_tab
        # print(tab + next_node.type_string)
        if not next_node.is_leaf():
            production_list.append(next_node.production)
            for child in reversed(next_node.children):
                stack.append((next_tab, child))
        else:
            production_list.append(LeafToken(type_id=next_node.type_id, type_string=next_node.type_string,
                                             value=next_node.value))
    return production_list


class MonitoredParser(object):
    def __init__(self,
                 lex_optimize=True,
                 lexer=BufferedCLex,
                 lextab='pycparser.lextab',
                 yacc_optimize=True,
                 yacctab='pycparser.yacctab',
                 yacc_debug=False,
                 taboutputdir='',
                 ):
        self._lex_optimize = lex_optimize
        self._lexer = lexer
        self._lextab = lextab
        self._yacc_optimize = yacc_optimize
        self._yacctab = yacctab
        self._yacc_debug = yacc_debug
        self._taboutputdir = taboutputdir
        self._parser = self._new_parser()

    def _new_parser(self):
        parser = CParser()
        is_parse_fn = create_is_p_fn()
        parse_fn_tuple_list = filter(lambda x: is_parse_fn(x[0]) and x[0] != "p_error", inspect.getmembers(parser))
        production_vocabulary = get_all_c99_production_vocabulary()

        def patch_fn(fn, doc, name, production):
            def wrapper(parse_self, p):
                """
                :param parse_self:
                :type p: c_code_processer.pycparser.pycparser.ply.yacc.YaccProduction
                :return:
                """
                cached_p = p[1:]
                for i, right_id in enumerate(production.right_id, start=1):
                    p[i] = p[i] if production_vocabulary.is_ternimal(right_id) else p[i][0]

                res = fn(p)
                for i, cache in enumerate(cached_p, start=1):
                    p[i] = cache
                left_node = ProductionNode(production, p[0])
                # has_ternimal_node = False
                for i, (right_id, right) in enumerate(zip(production.right_id, production.right), start=1):
                    if production_vocabulary.is_ternimal(right_id):
                        value = p[i]
                        # print(value)
                        # has_ternimal_node = True
                        child_node = LeafParseNode(right,
                                                   value,
                                                   right_id)
                    else:
                        child_node = p[i][1]

                    child_node.parent_node = left_node
                    left_node[i-1] = child_node
                # if has_ternimal_node:
                #     print(left_node)
                #     for child in left_node.children:
                #         print(child)
                p[0] = (p[0], left_node)
                return res

            wrapper.__name__ = name
            wrapper.__doc__ = doc
            return wrapper

        for k, v in parse_fn_tuple_list:
            # print("{}:{}".format(k, v))
            productions = production_vocabulary.get_production_by_production_string(v.__doc__)
            for i, production in enumerate(productions):
                name = k if i == 0 else k+str(i)
                new_method = types.MethodType(patch_fn(v, str(production), name, production), parser)
                setattr(parser, name, new_method)
        parser.build(
            self._lex_optimize,
            self._lexer,
            self._lextab,
            self._yacc_optimize,
            self._yacctab,
            self._yacc_debug
        )
        return parser

    def _parse(self, text, filename='', debuglevel=0):
        return self._parser.parse(text, filename, debuglevel)

    def parse(self, text, filename='', debuglevel=0):
        """
        :param text: the code string
        :return: the ast
        """
        return self._parse(text, filename, debuglevel)[0]

    def parse_get_production_list_and_token_list(self, code):
        """
        :param code: the code string
        :return: the parse tree , the ast, the tokens
        """
        final_ast = self._parse(code)
        tokens = [Token(value=t[0].value, type=t[0].type) for t in self._parser.clex.tokens_buffer]
        return final_ast[1], final_ast[0], tokens

    # def __getattr__(self, item):
    #     return getattr(self._parser, item)


if __name__ == '__main__':

    code = ''' #include <math.h> 
 #define a b
 int max(int a,int b){ 
 if (a>b)return a; 
 else return b;} 
int main(void) { 
 int n,a,b,c; 
 scanf( "%d %d %d %d" ,&n,&a,&b,&c); 
 int da[n+1]; 
 for(int i=0;i<=n;i++){ 
 da[i]=0; 
 } 
 for(int i=1;i<=n;i++){ 
 int x=-1,y=-1,z=-1; 
 if(i>a){ 
 x=da[i-a]; 
 } 
 if(i>b){ 
 y=da[i-b]; 
 } 
 if(i>c){ 
 y=da[i-c]; 
 } 
 da[i]=max(max(x,y),z)+1; 
 } 
 printf( "%d" ,da[1]); 
 return 0; 
}'''
    # code = replace_include_with_blank(code)
    # code = replace_define_with_blank(code)
    # code = replace_hashtag_with_blank(code)
    print(code)

    # show_all_c99_productions()
    # print(get_all_c99_production_vocabulary())
    # monitor = MonitoredParser(lex_optimize=False,
    #             yacc_debug=True,
    #             yacc_optimize=False,
    #             yacctab='yacctab')
    # code = """
    #     int add(int a, int b)
    #     {
    #         return a+b*c;
    #     }
    #     """
    # # show_production_node(monitor.parse_get_production_list_and_token_list(code)[0])
    # # print("The second show")
    # # show_production_node(monitor.parse_get_production_list_and_token_list(code)[0])
    # # production_vocabulary = get_all_c99_production_vocabulary()
    # # print(production_vocabulary)
    # # print(production_vocabulary._match_terminal_node)
    # print("top_down_show")
    # for t in parse_tree_to_top_down_process(monitor.parse_get_production_list_and_token_list(code)[0]):
    #     print(t)


