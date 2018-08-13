from c_parser.buffered_clex import BufferedCLex
from c_parser.pycparser.pycparser import CParser
from c_parser.pycparser.pycparser.c_ast import Node, FileAST

from collections import namedtuple, defaultdict
import re
import types
import inspect

class AsrException(Exception):
    def __init__(self, p, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.p = p


class AstParser(CParser):
    # def p_offsetof_member_designator(self, p):
    #     try:
    #         super().p_offsetof_member_designator(p)
    #     except Exception as e:
    #         self._sub_parser_raise_exception()
    #
    # def _build_declarations(self, spec, decls, typedef_namespace=False):
    #     try:
    #         super()._build_declarations(spec, decls, typedef_namespace=typedef_namespace)
    #     except Exception as e:
    #         self._sub_parser_raise_exception()
    #
    # def _build_function_definition(self, spec, decl, param_decls, body):
    #     try:
    #         super()._build_function_definition(spec, decl, param_decls, body)
    #     except Exception as e:
    #         self._sub_parser_raise_exception()
    #
    # def p_struct_declaration_1(self, p):
    #     try:
    #         super().p_struct_declaration_1(p)
    #     except Exception as e:
    #         self._sub_parser_raise_exception()
    #
    # def _parse_error(self, msg, coord):
    #     self._sub_parser_raise_exception()
    #
    # def _pop_scope(self):
    #     if len(self._scope_stack) > 1:
    #         self._sub_parser_raise_exception()
    #     self._scope_stack.pop()
    #
    # def p_error(self, p):
    #     self._sub_parser_raise_exception()

    def _sub_parser_raise_exception(self):
        raise AsrException([t.value for t in self.cparser.symstack[1:]])


def load_ast_parser():
    parser = AstParser()
    members = inspect.getmembers(parser)
    pattern = re.compile(r'__.*__')
    p_pattern = re.compile(r"p_.*")

    def patch_fn(f, name, doc):
        def wrapper_(parse_self, *args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                parse_self._sub_parser_raise_exception()

        def wrapper_p(parse_self, p):
            try:
                return f(p)
            except Exception as e:
                parse_self._sub_parser_raise_exception()

        if p_pattern.match(name):
            wrapper = wrapper_p
        else:
            wrapper = wrapper_

        wrapper.__name__ = name
        wrapper.__doc__ = doc
        return wrapper

    for k, v in filter(lambda x: pattern.match(x[0]) is None and x[0] != '_sub_parser_raise_exception' and x[0] != 'build',
                       members):
        new_method = types.MethodType(patch_fn(v, k, v.__doc__), parser)
        setattr(parser, k, new_method)
    parser.build(lexer=BufferedCLex)
    return parser


def ast_parse(code):
    if getattr(ast_parse, "parser", None) is None:
        ast_parse.parser = load_ast_parser()
    c_parser = ast_parse.parser
    try:
        ast = c_parser.parse(code)
        ast = [ast]
    except AsrException as e:
        ast = e.p
    return ast, [t[0] for t in c_parser.clex.tokens_buffer]


Coordinate = namedtuple("Coordinate", ["x", "y"])


class CodeGraph(object):
    def __init__(self, tokens, ast_list):
        self._tokens = tokens
        self._code_legth = len(tokens)
        self._pos_to_id_dict = self._generate_position_to_id(tokens)
        self._ast_list = ast_list
        self._link_tuple_list = [] # a tuple list (id1, id2, link_name)
        self._add_same_identifier_link()
        self._graph_ = [t.value for t in tokens]
        self._graph_.append("<Delimiter>")
        self._max_id = len(self._graph_)
        self._name_pattern = re.compile(r'(.+)\[\d+\]')
        self._parse_ast_list(ast_list)

    def _add_same_identifier_link(self):
        identifier_pos_map = defaultdict(list)
        for i, token in enumerate(self._tokens):
            if token.type == 'ID':
                identifier_pos_map[token.value].append(i)
        for k, v in identifier_pos_map.items():
            for i in range(len(v)):
                for j in range(i+1, len(v)):
                    self._add_link(v[i], v[j], "same_name_link")

    def _parse_ast_list(self, ast_list):
        for t in ast_list:
            if isinstance(t, list):
                self._parse_ast_list(t)
            elif isinstance(t, dict):
                self._parse_ast_list(t.values())
            elif isinstance(t, FileAST):
                for _, n in t.children():
                    self._parse_ast(n, self._new_node(n))
            elif isinstance(t, Node):
                self._parse_ast(t, self._new_node(t))

    @property
    def code_length(self):
        return self._code_legth

    @property
    def graph_length(self):
        return self._max_id

    @property
    def graph(self):
        """
        :return: (the graph node list(type is str),
                the link in the graph(a tuple list(node1_id:int, node2_id:int, link_type:str)))
        """
        return self._graph_, self._link_tuple_list

    def _generate_position_to_id(self, tokens):
        res = {}
        for i, token in enumerate(tokens):
            res[Coordinate(x=token.lineno, y=token.lexpos)] = i
        return res

    def _get_token_id_by_position(self, coord):
        # if coord is None:
        #     return None
        key = Coordinate(coord.line, coord.column)
        if key in self._pos_to_id_dict:
            return self._pos_to_id_dict[key]
        else:
            return None

    def _new_node(self, node):
        res = self._max_id
        node_name = type(node).__name__
        self._graph_.append(node_name)
        self._max_id += 1
        return res

    def _add_link(self, a, b, link_type="node_map"):
        if a is not None and b is not None:
            self._link_tuple_list.append((a, b, link_type))

    def _parse_name(self, name):
        m = self._name_pattern.match(name)
        if m is not None:
            return m.group(1)
        return name

    def _parse_ast(self, ast_node, node_id):
        token_id = self._get_token_id_by_position(ast_node.coord)
        self._add_link(node_id, token_id)
        for child_name, child_node in ast_node.children():
            child_name = self._parse_name(child_name)
            n_id = self._new_node(child_node)
            self._add_link(node_id, n_id, child_name)
            self._parse_ast(child_node, n_id)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        res = "The graph nodes:{}\n".format(" ".join(self._graph_))
        res += "\n".join(["{}:{}->{}:{} type:{}".format(link[0], self._graph_[link[0]], link[1], self._graph_[link[1]],
                                                       link[2]) for link in self._link_tuple_list])
        return res


def parse_ast_code_graph(token_list):
    ast, tokens = ast_parse("\n"+" ".join(token_list))
    return CodeGraph(tokens, ast)


if __name__ == '__main__':
    # load_ast_parser()
    code1 = """
        int add(int a,int b)
            return a+b;
        }
        """
    # code1 = """
    # long * memarray [ 3 ] ;
    # long getways ( int x , int m ) {
    #     int a , b , c ;
    #     static int sum = 0 ;
    #     if ( x == 0 ) {
    #         return 0 ;
    #         static ;
    #     }
    #     if ( x > 0 ) {
    #         a = x / 5 ;
    #         b = x - a ;
    #         c = ( x - a ) % 3 printf ( "%d%d%d" , a , b , c ) ;
    #         return getways ( x - 1 , m ) ;
    #     }
    # }
    #
    # int main ( ) {
    #     a = x / 5 ;
    #     b = x - a ;
    #     c = ( x - a ) % 3 ;
    #     printf ( "%d%d%d" , a , b , c ) ;
    #     return 0 ;
    # }
    # """
    ast, tokens = ast_parse(code1)
    print(ast)
    print(CodeGraph(tokens, ast))
    # code2 = """
    # int add(int a,int b){
    #     return a+b;
    # }
    # """
    # ast, tokens = ast_parse(code2)
    # print(ast)
    # print(CodeGraph(tokens, ast))

