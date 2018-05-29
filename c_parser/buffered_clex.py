from c_parser.pycparser.pycparser.c_lexer import CLexer
from c_parser.pycparser.pycparser.ply.lex import TOKEN
from common.util import maintain_function_co_firstlineno


class BufferedCLex(CLexer):
    def __init__(self, error_func, on_lbrace_func, on_rbrace_func, type_lookup_func):
        super().__init__(error_func, on_lbrace_func, on_rbrace_func, type_lookup_func)
        self._tokens_buffer = []
        self._tokens_index = 0

    @property
    def tokens_buffer(self):
        return self._tokens_buffer

    def token(self):
        if self._tokens_index < len(self._tokens_buffer):
            self.last_token = self._tokens_buffer[self._tokens_index][0]
            self.filename = self._tokens_buffer[self._tokens_index][1]
            self._tokens_index += 1
        else:
            self.last_token = None

        if self.last_token is not None:
            if self.last_token.type == 'LBRACE':
                self.on_lbrace_func()
            elif self.last_token.type == 'RBRACE':
                self.on_rbrace_func()
            elif self.last_token.type == 'ID' and self.type_lookup_func(self.last_token.value):
                self.last_token.type = 'TYPEID'

        return self.last_token

    def _all_tokens(self):
        def token():
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                else:
                    yield (tok, self.filename)
        return list(token())

    def input(self, text):
        super().input(text)
        self._tokens_buffer = self._all_tokens()
        # print(self._tokens_buffer)
        self._tokens_index = 0
        self.filename = ''

    @TOKEN(r'\}')
    @maintain_function_co_firstlineno(CLexer.t_LBRACE)
    def t_RBRACE(self, t):
        return t

    @TOKEN(r'\{')
    @maintain_function_co_firstlineno(CLexer.t_RBRACE)
    def t_LBRACE(self, t):
        return t

    @TOKEN(CLexer.identifier)
    @maintain_function_co_firstlineno(CLexer.t_ID)
    def t_ID(self, t):
        t.type = self.keyword_map.get(t.value, "ID")
        return t
