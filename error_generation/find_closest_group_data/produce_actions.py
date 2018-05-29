


# -------------------- calculate the recovery actions between two code -------------------------------- #

def get_action(matrix, i, j, a_token, b_token, left_action_fn, top_action_fn, left_top_action_fn, equal_fn, value_fn=lambda x: x):
    if i == 0 and j == 0:
        return None, -1, -1
    if i == 0:
        action = left_action_fn(i, j, a_token, b_token, value_fn)
        j -= 1
        return action, i, j
    if j == 0:
        action = top_action_fn(i, j, a_token, b_token, value_fn)
        i -= 1
        return action, i, j

    bias = 1
    if equal_fn(a_token, b_token):
        bias = 0

    # print('one: ', i, j, bias)
    # print('{} {} \n{} {}'.format(matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1], matrix[i][j]))

    if matrix[i][j] == (matrix[i-1][j-1] + bias):
        # do left_top
        action = left_top_action_fn(matrix, i, j, a_token, b_token, value_fn)
        i -= 1
        j -= 1
    elif matrix[i][j] == (matrix[i-1][j] + 1):
        # do top
        action = top_action_fn(i, j, a_token, b_token, value_fn)
        i -= 1
    elif matrix[i][j] == (matrix[i][j-1] + 1):
        #do left
        action = left_action_fn(i, j, a_token, b_token, value_fn)
        j -= 1
    else:
        print('get action position error')
        return None, None, None

    return action, i, j


def cal_action_list(matrix, a_tokens, b_tokens, left_action_fn, top_action_fn, left_top_action_fn, equal_fn=lambda x, y: x == y, value_fn=lambda x: x):
    """
    calculate action list from b tokens to a tokens
    :param matrix:
    :param a_tokens:
    :param b_tokens:
    :param left_action_fn:
    :param top_action_fn:
    :param left_top_action_fn:
    :param equal_fn:
    :param value_fn:
    :return:
    """
    len_a = len(a_tokens)
    len_b = len(b_tokens)

    action_list = []
    i = len_a
    j = len_b
    while i >= 0 and j >= 0:
        a_token = a_tokens[i-1] if i > 0 else None
        b_token = b_tokens[j-1] if j > 0 else None
        # if (action_is_list(a_token) or action_is_list(b_token)) and not equal_fn(a_token, b_token):
        #     print('in check action, {}, {}'.format(a_token, b_token))
        #     return None
        action, i, j = get_action(matrix, i, j, a_token, b_token, left_action_fn, top_action_fn, left_top_action_fn, equal_fn, value_fn)
        if action is not None:
            action_list = action_list + [action]
        if i is None:
            return None

    return action_list
