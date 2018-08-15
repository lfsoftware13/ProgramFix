from common.analyse_include_util import check_include_between_two_code
from common.constants import CACHE_DATA_PATH
from common.action_constants import ActionType
from common.util import init_code, check_ascii_character, disk_cache
from error_generation.find_closest_group_data.levenshtenin_token_level import levenshtenin_distance
from error_generation.find_closest_group_data.produce_actions import cal_action_list
from database.database_util import create_table, insert_items


def generate_equal_fn(token_value_fn=lambda x:x):
    def equal_fn(x, y):
        x_val = token_value_fn(x)
        y_val = token_value_fn(y)
        return x_val == y_val
    return equal_fn


def get_token_value(x):
    val = x.value
    if isinstance(val, list):
        val = ''.join(val)
    return val

a_tokenize_error_count = 0
ac_df_length_error = 0
distance_series_error = 0

def find_closest_token_text(one, ac_df, max_distance=None):
    global a_tokenize_error_count, ac_df_length_error, distance_series_error
    equal_fn = generate_equal_fn(get_token_value)
    a_tokenize = one['tokenize']
    a_code = one['code']
    ac_df = ac_df[ac_df['code'].map(lambda x: check_include_between_two_code(a_code, x))]

    if a_tokenize is None or len(ac_df) == 0:
        one['similar_code'] = ''
        one['action_list'] = []
        one['distance'] = -1
        one['similar_id'] = ''
        # print('a_tokenize is None {}, and len ac_df is {}'.format(type(a_tokenize), len(ac_df)))
        if a_tokenize is None:
            a_tokenize_error_count += 1
        elif len(ac_df) == 0:
            ac_df_length_error += 1
        return one
    cal_distance_fn = lambda x: levenshtenin_distance(a_tokenize, x, equal_fn=equal_fn, max_distance=max_distance)[0]
    distance_series = ac_df['tokenize'].map(cal_distance_fn)
    distance_series = distance_series[distance_series >= 0]
    if max_distance is not None and max_distance >= 1:
        distance_series = distance_series[distance_series < max_distance]
    if len(distance_series.index) <= 0:
        one['similar_code'] = ''
        one['action_list'] = []
        one['distance'] = -1
        one['similar_id'] = ''
        # print('distance series len is {}'.format(len(distance_series)))
        distance_series_error += 1
        return one
    min_id = distance_series.idxmin()
    min_value = distance_series.loc[min_id]

    b_tokenize = ac_df['tokenize'].loc[min_id]
    try:
        dis, matrix = levenshtenin_distance(a_tokenize, b_tokenize, equal_fn=equal_fn)
        if dis == -1:
            one['similar_code'] = ''
            one['action_list'] = []
            one['distance'] = -1
            one['similar_id'] = ''
            return one
        action_list = cal_action_list(matrix, a_tokenize, b_tokenize, left_move_action, top_move_action, left_top_move_action, equal_fn, get_token_value)
    except Exception as e:
        print(e)
        print('problem_user_id: ', one['problem_user_id'])
        one['similar_code'] = ''
        one['action_list'] = []
        one['distance'] = -1
        one['similar_id'] = ''
        return one
    one['distance'] = min_value
    one['similar_code'] = ac_df['code'].loc[min_id]
    one['action_list'] = action_list
    one['similar_id'] = ac_df['id'].loc[min_id]
    return one


def calculate_distance_and_action_between_two_code(error_tokenize, ac_tokenize, max_distance=None, get_value=None):
    get_token_value_fn = get_token_value
    if get_value is not None:
        get_token_value_fn = get_value
    equal_fn = generate_equal_fn(get_token_value_fn)
    distance, matrix = levenshtenin_distance(error_tokenize, ac_tokenize, equal_fn=equal_fn, max_distance=max_distance)
    if max_distance is not None and (distance < 0 or distance >= max_distance):
        distance = -1
        action_list = []
        return distance, action_list
    action_list = cal_action_list(matrix, error_tokenize, ac_tokenize, left_move_action, top_move_action,
                                  left_top_move_action, equal_fn, get_token_value_fn)
    return distance, action_list


def left_move_action(i, j, a_token, b_token, value_fn=lambda x: x):
    action = {'act_type': ActionType.DELETE, 'from_char': value_fn(b_token), 'to_char': '', 'token_pos': j-1}
    return action


def top_move_action(i, j, a_token, b_token, value_fn=lambda x: x):
    action = {'act_type': ActionType.INSERT_BEFORE, 'from_char': '', 'to_char': value_fn(a_token), 'token_pos': j}
    return action


def left_top_move_action(matrix, i, j, a_token, b_token, value_fn=lambda x: x):
    if matrix[i][j] == matrix[i-1][j-1]:
        return None
    action = {'act_type': ActionType.CHANGE, 'from_char': value_fn(b_token), 'to_char': value_fn(a_token), 'token_pos': j-1}
    return action


def recovery_code(tokens, action_list):
    # action_list.reverse()

    for act in action_list:
        act_type = act['act_type']
        pos = act['token_pos']
        from_char = act['from_char']
        to_char = act['to_char']
        if act_type == ActionType.INSERT_BEFORE:
            tokens = tokens[0:pos] + [to_char] + tokens[pos:]
        elif act_type == ActionType.DELETE:
            tokens = tokens[0:pos] + tokens[pos+1:]
        elif act_type == ActionType.CHANGE:
            tokens = tokens[0:pos] + [to_char] + tokens[pos+1:]
        else:
            print('action type error: {}'.format(act_type))
    return tokens


def create_id(one):
    user = one['user_id']
    problem = one['problem_name']
    return problem + '_' + user


@disk_cache(basename='init_c_code', directory=CACHE_DATA_PATH)
def init_c_code(data_df):
    data_df['problem_user_id'] = data_df.apply(create_id, axis=1, raw=True)
    data_df['code'] = data_df['code'].map(init_code)
    print('data length before check ascii: {}'.format(len(data_df)))
    data_df = data_df[data_df['code'].map(check_ascii_character)]
    print('data length after check ascii: {}'.format(len(data_df)))
    data_df = data_df[data_df['code'].map(lambda x: x != '')]
    return data_df


def filter_repeat_ids(data_df, problem_user_ids):
    data_df = data_df[data_df['problem_user_id'].map(lambda x: False if x in problem_user_ids else True)]
    return data_df


def save_train_data(error_df_list, ac_df_list, db_path, table_name, transform_fn):
    create_table(db_path, table_name)

    def trans(error_df):
        res = [transform_fn(row) for index, row in error_df.iterrows()]
        return res

    error_items_list = [trans(error_df) for error_df in error_df_list]
    for error_items in error_items_list:
        insert_items(db_path, table_name, error_items)
    # ac_items_list = [list(ac_df.apply(transform_data_list, raw=True, axis=1)) for ac_df in ac_df_list]
    # for ac_items in ac_items_list:
    #     insert_items(TRAIN_DATA_DBPATH, ACTUAL_C_ERROR_RECORDS, ac_items)







