from c_parser.pycparser.pycparser.ply.lex import LexToken
from common.action_constants import ActionType

import numpy as np


def generate_token_action(operations, tokens):
    """
    generate action list according to mark code object and operations produced by error generation
    :param marked_code:
    :param buffered_lexer:
    :param operations:
    :param tokens:
    :return:
    """
    token_actions = [[] for i in range(len(tokens) + 1)]
    operations = sorted(operations, key=position_weight_value, reverse=True)
    bias_list = cal_operations_bias(operations)
    for ope, bias in zip(operations, bias_list):
        action_type, position, text = ope
        if isinstance(action_type, int):
            action_type = ActionType(action_type)
        position += bias
        action_position = position + 1      # add a blank action in front of the code because insert after
        # print('action_position: {}, position: {}, bias: {}, token length: {}'.format(action_position, position, bias, len(tokens)))
        action_position = int(action_position)
        if action_type is ActionType.INSERT_BEFORE:
            token_actions = token_actions[:action_position] + [[]] + token_actions[action_position:]
            token_actions[action_position] += [(ActionType.DELETE, None)]

            tmp_token = init_LexToken(text)
            tokens = modify_lex_tokens_offset(tokens, action_type, position, tmp_token)
        elif action_type is ActionType.INSERT_AFTER:
            token_actions = token_actions[:action_position + 1] + [[]] + token_actions[action_position + 1:]
            token_actions[action_position + 1] += [(ActionType.DELETE, None)]

            tmp_token = init_LexToken(text)
            tokens = modify_lex_tokens_offset(tokens, action_type, position, tmp_token)
        elif action_type is ActionType.DELETE:
            tmp_token = tokens[position]
            if action_position == 0:
                tmp_pass_actions = filter_action_types(token_actions[action_position], [ActionType.INSERT_BEFORE])
                token_actions[action_position + 1] += tmp_pass_actions
                token_actions[action_position + 1] += [(ActionType.INSERT_BEFORE, tmp_token)]
            else:
                tmp_pass_actions = filter_action_types(token_actions[action_position], [ActionType.INSERT_AFTER])
                token_actions[action_position - 1] += tmp_pass_actions
                token_actions[action_position - 1] += [(ActionType.INSERT_AFTER, tmp_token)]
            token_actions = token_actions[:action_position] + token_actions[action_position + 1:]
            tokens = modify_lex_tokens_offset(tokens, action_type, position)
        elif action_type is ActionType.CHANGE:
                tmp_token = tokens[position]
                token_actions[action_position] += [(ActionType.CHANGE, tmp_token)]
                tmp_modify_token = init_LexToken(text)
                tokens = modify_lex_tokens_offset(tokens, action_type, position, tmp_modify_token)

    return tokens, token_actions


def cal_operations_bias(operations):
    def action_bias_value(action_type):
        if action_type is ActionType.INSERT_BEFORE or action_type is ActionType.INSERT_AFTER:
            return 1
        elif action_type is ActionType.DELETE:
            return -1
        return 0

    ope_value_fn = lambda ope, cur_ope: action_bias_value(ope[0]) if position_weight_value(ope) < position_weight_value(cur_ope) else 0
    operations_value_fn = lambda operations, i: int(np.sum([ope_value_fn(ope, operations[i]) for ope in operations[:i]]))

    operation_bias_list = [operations_value_fn(operations, i) for i in range(len(operations))]
    return operation_bias_list


def position_weight_value(ope):
    action_type = ope[0]
    position = ope[1]
    if action_type is ActionType.INSERT_BEFORE:
        return position - 0.5
    elif action_type is ActionType.INSERT_AFTER:
        return position + 0.5
    return position


def init_LexToken(value, type=None, lineno=-1, lexpos=-1):
    token = LexToken()
    token.value = value
    token.type = type
    token.lineno = lineno
    token.lexpos = lexpos
    return token


def filter_action_types(actions, types):
    res = []
    for act in actions:
        if act[0] in types:
            res += [act]
    return res


# ------------------------------------------ produce new tokens by action ------------------------------- #

def modify_lex_tokens_offset(ori_tokens: list, action_type, position, token=None):
    """
    Modify the lex token list according to action object. This function will also modify offset of tokens. Lineno will
    not change.
    Action: {action_type, action_position, action_value}
    :param ori_tokens:
    :param action_type:
    :param position:
    :param token
    :return:
    """
    if isinstance(action_type, int):
        action_type = ActionType(action_type)

    if position < 0 or (action_type == ActionType.INSERT_BEFORE and position > len(ori_tokens)) \
            or (action_type != ActionType.INSERT_BEFORE and position >= len(ori_tokens)):
        raise Exception('action position error. ori_tokens len: {}, action_type: {}, position: {}\n ' +
                        'token.type: {}, token.value: {}'.format(len(ori_tokens), action_type, position,
                                                                 token.type, token.value))

    new_tokens = ori_tokens
    if isinstance(new_tokens, tuple):
        new_tokens = list(new_tokens)

    if action_type is not ActionType.INSERT_BEFORE and action_type is not ActionType.INSERT_AFTER:
        new_tokens = new_tokens[:position] + new_tokens[position+1:]
        bias = 0 - len(ori_tokens[position].value) + 1
        new_tokens = modify_bias(new_tokens, position, bias)

    if action_type is not ActionType.DELETE:
        token, token_index = set_token_position_info(new_tokens, action_type, position, token)
        new_tokens = new_tokens[:token_index] + [token] + new_tokens[token_index:]
        bias = len(token.value) + 2
        new_tokens = modify_bias(new_tokens, token_index + 1, bias)
    return new_tokens


def set_token_position_info(tokens, action_type, position, token):
    if (action_type is ActionType.INSERT_BEFORE or action_type is ActionType.CHANGE) and position == len(tokens):
        position -= 1
        action_type = ActionType.INSERT_AFTER
    if position < len(tokens) and action_type is not ActionType.INSERT_AFTER:
        according_token = tokens[position]
        token = set_token_line_pos_accroding_before(according_token, token)
    else:
        if action_type is ActionType.INSERT_BEFORE:
            position -= 1
        according_token = tokens[position]
        token = set_token_line_pos_accroding_after(according_token, token)
        position += 1
    return token, position


def set_token_line_pos_accroding_before(according_token, token):
    token.lineno = according_token.lineno
    token.lexpos = according_token.lexpos + 1
    return token


def set_token_line_pos_accroding_after(according_token, token):
    token.lineno = according_token.lineno
    token.lexpos = according_token.lexpos + len(according_token.value) + 1
    return token


def modify_bias(tokens, position, bias):
    for tok in tokens[position:]:
        tok.lexpos += bias
    return tokens
