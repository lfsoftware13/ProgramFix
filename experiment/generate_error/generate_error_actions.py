from error_generation.find_closest_group_data.levenshtenin_token_level import levenshtenin_distance
from error_generation.find_closest_group_data.produce_actions import cal_action_list

CHANGE = 0
INSERT = 1
DELETE = 2

def left_move_action(i, j, a_token, b_token, value_fn=lambda x: x):
    action = {'act_type': DELETE, 'from_char': value_fn(b_token), 'to_char': '', 'token_pos': j-1}
    return action


def top_move_action(i, j, a_token, b_token, value_fn=lambda x: x):
    action = {'act_type': INSERT, 'from_char': '', 'to_char': value_fn(a_token), 'token_pos': j}
    return action


def left_top_move_action(matrix, i, j, a_token, b_token, value_fn=lambda x: x):
    if matrix[i][j] == matrix[i-1][j-1]:
        return None
    action = {'act_type': CHANGE, 'from_char': value_fn(b_token), 'to_char': value_fn(a_token), 'token_pos': j-1}
    return action


def recovery_code(tokens, action_list):
    # action_list.reverse()

    for act in action_list:
        act_type = act['act_type']
        pos = act['token_pos']
        from_char = act['from_char']
        to_char = act['to_char']
        if act_type == INSERT:
            tokens = tokens[0:pos] + [to_char] + tokens[pos:]
        elif act_type == DELETE:
            tokens = tokens[0:pos] + tokens[pos+1:]
        elif act_type == CHANGE:
            tokens = tokens[0:pos] + [to_char] + tokens[pos+1:]
        else:
            print('action type error: {}'.format(act_type))
    return tokens


def generate_actions_from_ac_to_error_by_code(error_ids, ac_ids, max_distance=10):
    dis, matrix = levenshtenin_distance(error_ids, ac_ids, max_distance=max_distance)
    if dis > max_distance:
        return dis, None
    action_list = cal_action_list(matrix, error_ids, ac_ids, left_move_action, top_move_action,
                                  left_top_move_action)
    return dis, action_list
