import json
import logging
import multiprocessing as mp
import os
import queue
# from scripts.scripts_util import initLogging
import random
import time

from common.token_operation_util import generate_token_action
from common.pycparser_util import tokenize_by_clex_fn
from error_generation.generation_error.action_mapitem import ACTION_MAPITEM, ERROR_CHARACTER_MAPITEM
from common.constants import COMPILE_TMP_PATH, RANDOM_C_ERROR_RECORDS, FAKE_C_COMPILE_ERROR_DATA_DBPATH, \
    COMMON_C_ERROR_RECORDS, COMMON_DEEPFIX_ERROR_RECORDS
from config import FAKE_DEEPFIX_ERROR_DATA_DBPATH
from error_generation.generation_error.error_action_reducer import create_error_action_fn
from read_data.read_experiment_data import read_deepfix_ac_data
from read_data.read_filter_data_records import read_distinct_problem_user_ac_c_records_filter_error_code, \
    read_deepfix_ac_records
from database.database_util import insert_items, create_table, run_sql_select_statment
from common.util import init_code, compile_c_code_by_gcc, build_code_string_from_lex_tokens
from common.analyse_include_util import replace_include_with_blank, extract_include, analyse_include_line_no, \
    extract_include_name

preprocess_logger = logging.getLogger('code_preprocess')
# 设置logger的level为DEBUG

preprocess_logger.setLevel(logging.DEBUG)
preprocess_logger.__setattr__('propagate', False)
# 创建一个输出日志到控制台的StreamHandler
# hdr = logging.StreamHandler()
hdr = logging.FileHandler("log/generate_deepfix_common_error.log")
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
hdr.setFormatter(formatter)
# 给logger添加上handler
preprocess_logger.addHandler(hdr)

compile_max_count = 1
error_max_count = 1
failed_max_count = 1

current_table_name = COMMON_DEEPFIX_ERROR_RECORDS
db_name = FAKE_DEEPFIX_ERROR_DATA_DBPATH

def preprocess():
    # initLogging()
    preprocess_logger.info("Start Read Code Data")
    code_df = read_deepfix_ac_data()
    preprocess_logger.info("Code Data Read Finish. Total: {}".format(code_df.shape[0]))
    que_read = mp.Queue()
    que_write = mp.Queue()

    create_table(db_full_path=db_name, table_name=current_table_name)
    pros = []
    for i in range(6):
        pro = mp.Process(target=make_fake_code, args=(que_read, que_write, i))
        pro.start()
        pros.append(pro)
    save_pro = mp.Process(target=save_fake_code, args=(que_write, code_df.shape[0]))
    save_pro.start()

    count = 0
    ids = []
    items = []
    for index, row in code_df.iterrows():
        count += 1
        # item = create_codeforce_item(row)
        item = create_deepfix_item(row)
        items.append(item)

        ids.append(item['problem_user_id'])

        if len(ids) == 10000:
            push_code_to_queue(que_read, ids, items)
            preprocess_logger.info('Total Preprocess {}'.format(count))
            ids = []
            items = []

    push_code_to_queue(que_read, ids, items)
    preprocess_logger.info('Total Preprocess {}'.format(count))

    for p in pros:
        p.join()
    save_pro.join()


def create_codeforce_item(row):
    # for codeforce code
    item = {'try_count': 0}
    item['id'] = row['id']
    item['submit_url'] = row['submit_url']
    item['problem_id'] = row['problem_id']
    item['user_id'] = row['user_id']
    item['problem_user_id'] = row['problem_user_id']
    item['originalcode'] = row['code'].replace('\ufeff', '').replace('\u3000', ' ')
    return item


def create_deepfix_item(row):
    # for deepfix code
    item = {'try_count': 0}
    item['id'] = row['code_id']
    item['submit_url'] = ''
    item['problem_id'] = row['problem_id']
    item['user_id'] = row['user_id']
    item['problem_user_id'] = row['problem_id'] + '_' +row['user_id']
    item['originalcode'] = row['code'].replace('\ufeff', '').replace('\u3000', ' ')
    return item


def push_code_to_queue(que, ids, items):
    ids = ["'" + t + "'" for t in ids]
    result_list = run_sql_select_statment(db_full_path=db_name, table_name=current_table_name, sql_name='find_distinct_problem_user_id')
    ids_repeat = [row[0] for row in result_list]
    count = 0
    for it in items:
        if it['problem_user_id'] not in ids_repeat:
            count += 1
            que.put(it)
        else:
            que.put(None)
    preprocess_logger.info('Preprocess {} code in {}'.format(count, len(ids)))

def make_fake_code(que_read:mp.Queue, que_write:mp.Queue, ind:int):
    preprocess_logger.info('Start Make Fake Code Process {}'.format(ind))
    tmp_code_file_path = os.path.join(COMPILE_TMP_PATH, 'code'+str(ind)+'.c')
    timeout_count = 0
    count = 0
    success_count = 0
    err_count = 0
    fail_count = 0
    repeat_count = 0
    tokenize_fn = tokenize_by_clex_fn()
    while True:
        if timeout_count >= 5:
            break

        if count % 10 == 0:
            preprocess_logger.info("Process {} | count: {} | error_count: {} | fail_count: {} | repeat_count: {}".format(ind, count, err_count, fail_count, repeat_count))

        try:
            item = que_read.get(timeout=600)
        except queue.Empty:
            timeout_count += 1
            continue
        except TimeoutError:
            timeout_count += 1
            continue

        timeout_count = 0
        count += 1
        if not item:
            repeat_count += 1
            que_write.put(None)
            continue

        # item['originalcode'] = item['originalcode'].replace('\ufeff', '').replace('\u3000', ' ')

        try:
            before_code, after_code, action_maplist, error_character_maplist, error_count = preprocess_code(item['originalcode'], cpp_file_path=tmp_code_file_path, tokenize_fn=tokenize_fn)
        except Exception as e:
            preprocess_logger.info('error info: ' + str(e))
            before_code = None
            after_code = None
            action_maplist = None
            error_character_maplist = None
            error_count = 1

        count += 1
        if before_code:
            success_count += 1
            item['ac_code'] = before_code
            item['code'] = after_code
            item['error_count'] = error_count
            error_list = list(map(lambda x: x.__dict__(), error_character_maplist))
            action_list = list(map(lambda x: x.__dict__(), action_maplist))
            item['error_character_maplist'] = error_list
            item['action_maplist'] = action_list
            que_write.put(item)
        else:
            item['try_count'] += 1
            if item['try_count'] < error_max_count:
                err_count += 1
                que_read.put(item)
            else:
                fail_count += 1
                que_write.put(None)

    preprocess_logger.info("Process {} | count: {} | error_count: {} | fail_count: {} | repeat_count: {}".format(ind, count, err_count, fail_count,  repeat_count))
    preprocess_logger.info('End Make Fake Code Process {}'.format(ind))


def save_fake_code(que:mp.Queue, all_data_count):
    create_table(db_full_path=db_name, table_name=current_table_name)
    que.qsize()
    preprocess_logger.info('Start Save Fake Code Process. all data count: {}'.format(all_data_count))
    count = 0
    error_count = 0
    param = []
    while True:
        if not que.empty() and count < all_data_count:
            try:
                preprocess_logger.info('before get item: {}'.format(count))
                item = que.get()
                preprocess_logger.info('after get item: {}'.format(count))
            except TypeError as e:
                preprocess_logger.info('Save get Type Error')
                error_count += 1
                continue
            count += 1
            if count % 1000 == 0:
                preprocess_logger.info('Total receive records: {}'.format(count))
            if not item:
                continue
            param.append(item)
            preprocess_logger.info('save data count: {}. current count: {}, Wait item: {}, Que is Empty: {}'.format(count, len(param), que.qsize(), que.empty()))
            if len(param) > 1000:
                preprocess_logger.info('Save {} recode. Total record: {}. error count: {}. Wait item: {}'.format(len(param), count, error_count, que.qsize()))
                insert_items(db_full_path=db_name, table_name=current_table_name, params=dict_to_list(param))
                param = []
        elif que.empty() and count >= all_data_count:
            break
        elif que.qsize() <= 0:
            time.sleep(1)
    preprocess_logger.info('Save {} recode. Total record: {}. error count: {}. Wait item: {}'.format(len(param), count, error_count, que.qsize()))
    insert_items(db_full_path=db_name, table_name=current_table_name, params=dict_to_list(param))
    preprocess_logger.info('End Save Fake Code Process')


def dict_to_list(param):
    param_list = []
    preprocess_logger.info('dict to list start: {}'.format(len(param)))
    for pa in param:
        item = []
        item.append(pa['id'])
        item.append(pa['submit_url'])
        item.append(pa['problem_id'])
        item.append(pa['user_id'])
        item.append(pa['problem_user_id'])
        item.append(pa['code'])
        item.append(pa['ac_code'])
        action_list = json.dumps(pa['action_maplist'])
        item.append(action_list)
        item.append(pa['error_count'])
        # error_list = json.dumps(pa['error_character_maplist'])
        # item.append(error_list)
        # error_list = list(map(lambda x: x.__dict__(), pa['error_character_maplist']))
        # action_list = list(map(lambda x: x.__dict__(), pa['action_character_list']))
        param_list.append(item)
    preprocess_logger.info('end dict to list: {}'.format(len(param_list)))
    return param_list


def preprocess_code(code, cpp_file_path=COMPILE_TMP_PATH, tokenize_fn=None):
    if not compile_c_code_by_gcc(code, cpp_file_path):
        return None, None, None, None, None
    code = init_code(code)
    if not compile_c_code_by_gcc(code, cpp_file_path):
        return None, None, None, None, None
    before_code = code
    after_code = before_code
    error_count_range = (1, 9)
    if tokenize_fn is None:
        tokenize_fn = tokenize_by_clex_fn()

    count = 0
    action_maplist = []
    error_character_maplist = []
    error_count = -1
    while compile_c_code_by_gcc(after_code, cpp_file_path):
        cod = before_code
        # cod = remove_blank(cod)
        # cod = remove_comments(cod)
        # cod = remove_blank_line(cod)
        count += 1
        # before_code = cod
        before_code, after_code, action_maplist, error_character_maplist, error_count = create_error_code(cod, error_count_range=error_count_range, tokenize_fn=tokenize_fn)
        if before_code is None:
            return None, None, None, None, None
        if count > compile_max_count:
            return None, None, None, None, None

    return before_code, after_code, action_maplist, error_character_maplist, error_count

CHANGE = 4
INSERT = 1
DELETE = 3
STAY = 5
FILL = 6

def fill_blank_to_error_code(error_character_maplist, ac_i, err_i):
    item = ERROR_CHARACTER_MAPITEM(act_type=FILL, from_char=' ', err_pos=err_i, ac_pos=ac_i)
    error_character_maplist.append(item)
    return error_character_maplist


def create_error_code(code, error_type_list=(5, 1, 4), error_count_range=(1, 5), tokenize_fn = None):
    code_without_include = replace_include_with_blank(code)
    include_lines = extract_include(code)
    include_line_nos = analyse_include_line_no(code, include_lines)

    try:
        if tokenize_fn is None:
            tokenize_fn = tokenize_by_clex_fn()
        code_tokens = tokenize_fn(code_without_include)
        if code_tokens is None or len(code_tokens) > 1000:
            # preprocess_logger.info('code tokens is None: {}'.format(code_without_include))
            preprocess_logger.info('code tokens is None')
            return None, None, None, None, None

    except Exception as e:
        preprocess_logger.info('tokenize code error.')
        return None, None, None, None, None

    error_count = random.randint(*error_count_range)
    action_maplist = create_multi_error(code_without_include, code_tokens, error_type_list, error_count)
    # action_mapposlist = list(map(lambda x: x.get_ac_pos(), action_maplist))
    error_character_maplist = []
    # ac_code_list = list(code)
    #
    # ac_i = 0
    # err_i = 0

    # def get_action(act_type, ac_pos):
    #     for i in action_maplist:
    #         if act_type == i.act_type and ac_pos == i.ac_pos:
    #             return i
    #     return None

    # for ac_i in range(len(ac_code_list)):
    # while ac_i < len(ac_code_list):
    #     if ac_i in action_mapposlist and get_action(act_type=DELETE, ac_pos=ac_i) != None:
    #         action = get_action(act_type=DELETE, ac_pos=ac_i)
    #         error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #         err_i += 1
    #
    #         action.err_pos = err_i
    #         ac_i += len(action.from_char)
    #
    #         error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #         err_i += 1
    #         continue
    #
    #     if ac_i in action_mapposlist and get_action(act_type=INSERT, ac_pos=ac_i) != None:
    #         action = get_action(act_type=INSERT, ac_pos=ac_i)
    #         error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #         err_i += 1
    #
    #         action.err_pos = err_i
    #         for i in range(len(action.to_char)):
    #             err_item = ERROR_CHARACTER_MAPITEM(act_type=INSERT, from_char=action.to_char[i], err_pos=err_i, ac_pos=ac_i)
    #             error_character_maplist.append(err_item)
    #             err_i += 1
    #
    #         error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #         err_i += 1
    #
    #     if ac_i in action_mapposlist and get_action(act_type=CHANGE, ac_pos=ac_i) != None:
    #         action = get_action(act_type=CHANGE, ac_pos=ac_i)
    #         error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #         err_i += 1
    #
    #         action.err_pos = err_i
    #         for i in range(len(action.to_char)):
    #             err_item = ERROR_CHARACTER_MAPITEM(act_type=CHANGE, from_char=action.to_char[i], err_pos=err_i, to_char=action.from_char, ac_pos=ac_i)
    #             err_i += 1
    #             error_character_maplist.append(err_item)
    #         ac_i += len(action.from_char)
    #
    #         error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #         err_i += 1
    #
    #     else:
    #         err_item = ERROR_CHARACTER_MAPITEM(act_type=STAY, from_char=code[ac_i], err_pos=err_i, to_char=code[ac_i],
    #                                            ac_pos=ac_i)
    #         err_i += 1
    #         error_character_maplist.append(err_item)
    #         ac_i += 1
    #
    # if ac_i in action_mapposlist and get_action(act_type=INSERT, ac_pos=ac_i) != None:
    #     action = get_action(act_type=INSERT, ac_pos=ac_i)
    #     error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #     err_i += 1
    #
    #     action.err_pos = err_i
    #     for i in range(len(action.to_char)):
    #         err_item = ERROR_CHARACTER_MAPITEM(act_type=INSERT, from_char=action.to_char[i], err_pos=err_i, ac_pos=ac_i)
    #         error_character_maplist.append(err_item)
    #         err_i += 1
    #
    #     error_character_maplist = fill_blank_to_error_code(error_character_maplist, ac_i, err_i)
    #     err_i += 1

    def convert_action_list_to_operation_tuple(one_action):
        val = None
        if one_action.act_type == INSERT or one_action.act_type == CHANGE:
            val = one_action.to_char
        elif one_action.act_type == DELETE:
            val = one_action.from_char
        return [one_action.act_type, one_action.token_pos, val]

    operation_list = [convert_action_list_to_operation_tuple(act) for act in action_maplist]
    error_tokens, _ = generate_token_action(operation_list, tokens=code_tokens)
    if error_tokens is None:
        return None, None, None, None, None
    error_code = build_code_string_from_lex_tokens(error_tokens)
    error_lines = error_code.split('\n')
    for name, line_no in zip(include_lines, include_line_nos):
        if error_lines[line_no].strip() == '':
            error_lines[line_no] = name
        else:
            # preprocess_logger.info('tokens: {}'.format(error_tokens))
            # preprocess_logger.info('code: {}'.format(error_code))
            # preprocess_logger.info('extract include: {}'.format(include_lines))
            preprocess_logger.info('extract include lineno: {}'.format(include_line_nos))
            preprocess_logger.info('add include error: {}'.format(error_lines[line_no]))
    error_code = '\n'.join(error_lines)
    # error_code = ''.join(list(map(lambda x: x.from_char, error_character_maplist)))

    return code, error_code, action_maplist, error_character_maplist, error_count


def create_multi_error(code, code_tokens, error_type_list=(5, 1, 4), error_count=1):
    # code_len = len(code)
    if len(error_type_list) != 3:
        return []

    action_maplist = []
    token_pos_list = []
    try_count = 0
    while len(action_maplist) < error_count and try_count < failed_max_count:
        error_action_fn = create_error_action_fn()
        # act_type, pos, token_pos, from_char, to_char = error_action_fn(code, code_tokens)
        action_tuple_list = error_action_fn(code, code_tokens)
        if action_tuple_list == None:
            try_count += 1
            continue
        without_insert_pos_list = [i[2] if i[0] != INSERT else -1 for i in action_tuple_list]
        token_pos_tmp_list = filter(lambda x: x != -1, without_insert_pos_list)
        # token_pos_tmp_list = [i[2] for i in action_tuple_list]
        while len(set(token_pos_tmp_list) & set(token_pos_list)) > 0 and try_count < failed_max_count:
            action_tuple_list = error_action_fn(code, code_tokens)
            if action_tuple_list == None:
                try_count += 1
                continue
            without_insert_pos_list = [i[2] if i[0] != INSERT else -1 for i in action_tuple_list]
            token_pos_tmp_list = filter(lambda x: x != -1, without_insert_pos_list)
            # token_pos_tmp_list = [i[2] for i in action_tuple_list]
            try_count += 1

        if try_count >= failed_max_count:
            break
        token_pos_list.extend(token_pos_tmp_list)
        for act in action_tuple_list:
            act_type, pos, token_pos, from_char, to_char = act
            action_item = ACTION_MAPITEM(act_type=act_type, ac_pos=pos, token_pos=token_pos, from_char=from_char, to_char=to_char)
            action_maplist.append(action_item)

    if try_count >= failed_max_count:
        preprocess_logger.info('action list is empty because max count')
        return []

    if len(action_maplist) > 9:
        preprocess_logger.info('action list is empty because action_list count: {}. error count: {}'.format(len(action_maplist), error_count))
        return []

    return action_maplist


# def create_error(code, error_type_list=(1, 1, 1), error_count=1):
#     code_len = len(code)
#     new_code = code
#     if len(error_type_list) != 3:
#         return code, -1
#     res = random.uniform(0, sum(error_type_list))
#     act_type = 0
#     act_type = act_type + 1 if res > error_type_list[0] else act_type
#     act_type = act_type + 1 if res > error_type_list[1] else act_type
# 
#     act_pos = -1
#     act_cha_sign = -1
#     if act_type == 0:
#         pos = random.randint(0, code_len-1)
#         new_code = new_code[:pos] + new_code[(pos + 1):]
#         act_pos = pos * 2
#         if code[pos] not in char_sign_dict.keys():
#             return None, None, None, None, None
#         act_cha_sign = char_sign_dict[code[pos]]
#     elif act_type == 1:
#         pos = random.randint(0, code_len)
#         cha = sign_char_dict[random.randint(0, len(sign_char_dict)-2)]
#         new_code = new_code[:pos] + cha +new_code[pos:]
#         act_pos = pos * 2 + 1
#         act_cha_sign = 96
#     elif act_type == 2:
#         pos = random.randint(0, code_len-1)
#         cha = sign_char_dict[random.randint(0, len(sign_char_dict)-2)]
#         new_code = new_code[:pos] + cha +new_code[(pos + 1):]
#         act_pos = pos * 2 + 1
#         if code[pos] not in char_sign_dict.keys():
#             return None, None, None, None, None
#         act_cha_sign = char_sign_dict[code[pos]]
# 
#     return code, new_code, act_type, act_pos, act_cha_sign


if __name__ == '__main__':
    preprocess()

