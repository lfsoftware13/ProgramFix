import errno
import functools
import itertools
import os
import pickle
import random
import re
import types
from multiprocessing import Pool
import typing
import hashlib

import copy
from typing import Iterator

import more_itertools
import sklearn
import pandas as pd
import sys
# import cytoolz as toolz
import toolz
import collections

import time

from sklearn.utils import shuffle
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from common.args_util import get_compile_pool
from common.logger import info
from common.new_tokenizer import tokenize
from config import num_processes


def make_dir(*path: str) -> None:
    """
    This method will recursively create the directory
    :param path: a variable length parameter
    :return:
    """
    path = os.path.join(*path)

    if not path:
        return

    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError("The path {} already exits but it is not a directory".format(path))
        return

    base, name = os.path.split(path)
    make_dir(base)
    if name:
        os.mkdir(path)


def format_dict_to_string(to_format_dict: dict) -> str:
    """
    :param to_format_dict: a dict to format
    :return:
    """

    def to_str(o):
        if is_sequence(o):
            return ''.join(to_str(t) for t in o)
        else:
            return str(o)
    # print(len('__'.join(to_str(a)+to_str(b) for a, b in to_format_dict.items())))
    return '__'.join(to_str(a)+to_str(b) for a, b in to_format_dict.items())


def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def disk_cache(basename, directory, method=False):
    """
    Function decorator for caching pickleable return values on disk. Uses a
    hash computed from the function arguments for invalidation. If 'method',
    skip the first argument, usually being self or cls. The cache filepath is
    'directory/basename-hash.pickle'.
    """
    directory = os.path.expanduser(directory)
    ensure_directory(directory)

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (tuple(args), tuple(kwargs.items()))
            # Don't use self or cls for the invalidation hash.
            if method and key:
                key = key[1:]
            filename = '{}-{}.pickle'.format(basename, data_hash(key))
            print("the cache name is {}".format(filename))
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print("load file from:{}".format(filepath))
                with open(filepath, 'rb') as handle:
                    return pickle.load(handle)
            result = func(*args, **kwargs)
            with open(filepath, 'wb') as handle:
                print("write cache to: {}".format(filepath))
                pickle.dump(result, handle)
            return result
        return wrapped

    return wrapper


def data_hash(key):

    def hash_value(hash_item):
        v = 0
        try:
            v = int(hashlib.md5(str(hash_item).encode('utf-8')).hexdigest(), 16)
        except Exception as e:
            print('error occur while hash item {} '.format(type(hash_item)))
        return v

    hash_val = 0
    key = list(more_itertools.flatten(key))
    for item in key:
        if isinstance(item, pd.DataFrame):
            serlist = [item.itertuples(index=False, name=None)]
            serlist = list(more_itertools.collapse(serlist))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, pd.Series):
            serlist = item.tolist()
            serlist = list(more_itertools.collapse(serlist))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, int) or isinstance(item, float) or isinstance(item, str):
            val = hash_value(item)
            hash_val += val
        elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
            serlist = list(more_itertools.collapse(item))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, dict):
            serlist = list(more_itertools.collapse(item.items()))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        else:
            print('type {} cant be hashed.'.format(type(item)))
    return str(hash_val)

# ================================================================
# multiprocess function
# ================================================================

def parallel_map(core_num, f, args):
    """
    :param core_num: the cpu number
    :param f: the function to parallel to do
    :param args: the input args
    :return:
    """

    with Pool(core_num) as p:
        r = p.map(f, args)
        return r

# ================================================================
# dict function
# ================================================================

def reverse_dict(d: dict) -> dict:
    """
    swap key and value of a dict
    dict(key->value) => dict(value->key)
    """
    return dict(map(reversed, d.items()))

# ================================================================
# sequence function
# ================================================================

def is_sequence(s):
    try:
        iterator = iter(s)
    except TypeError:
        return False
    else:
        if isinstance(s, str):
            return False
        return True


def convert_to_list(s):
    if is_sequence(s):
        return list(s)
    else:
        return [s]


def sequence_sum(itr):
    return sum(itr)

def padded_code_new(batch_code, fill_value):
    if not isinstance(batch_code, list):
        return batch_code
    elif not isinstance(batch_code[0], list):
        return batch_code

    batch_root = batch_code
    while True:
        if not isinstance(batch_root, list):
            return batch_code
        elif not isinstance(batch_root[0], list):
            return batch_code
        cur_fill_value = fill_value
        if isinstance(batch_root[0][0], list):
            cur_fill_value = []
        max_len = max(map(len, batch_root))
        for b in batch_root:
            while len(b) < max_len:
                b.append(cur_fill_value)
        # list(map(lambda x: list(more_itertools.padded(x, fillvalue=fill_value, n=max_len)), batch_root))

        tmp = []
        for son in batch_root:
            for s in son:
                tmp.append(s)
        batch_root = tmp

def padded(x, deepcopy=False, fill_value=0):
    import copy
    if deepcopy:
        x = copy.deepcopy(x)
    if not isinstance(x, list):
        return x
    elif isinstance(x[0], list):
        return padded_code_new(x, fill_value=fill_value)
    else:
        return x

def padded_to_length(x, length, fill_value):
    res = list(more_itertools.padded(x, fill_value, length))
    return res


def batch_holder(*data: typing.List, batch_size=32,):
    """
    :param data:
    :return:
    """
    def iterator():
        def one_epoch():
            i_data = list(map(lambda x: more_itertools.chunked(x, batch_size), data))
            return zip(*i_data)
        for i ,m in enumerate(more_itertools.repeatfunc(one_epoch, times=1)):
            for t in m:
                yield t

    return iterator

def dataset_holder(*args):
    def f():
        return args
    return f

def train_test_split(data, test_size):
    from sklearn.model_selection import train_test_split
    data = train_test_split(*data, test_size=test_size)

    d_len = len(data)
    train_data = [data[i] for i in range(0, d_len, 2)]
    test_data = [data[i] for i in range(1, d_len, 2)]
    return train_data, test_data

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def get_count_size(x):
    r = 0
    if hasattr(x, '__iter__') and not isinstance(x, (str, bytes, bytearray)):
        r += sum(get_count_size(t) for t in x)
    else:
        r += 1

    return r


def unique_adjacent(seq: typing.Iterator):
    pre = next(seq)
    yield pre
    for token in seq:
        if token == pre:
            continue
        else:
            pre = token
            yield pre


def group_df_to_grouped_list(data_df, groupby_key):
    grouped = data_df.groupby(groupby_key)
    group_list = []
    for name, group in grouped:
        group_list += [group]
    return group_list


def filter_length(df, limit_length, tokenize_fn, code_key='similar_code'):
    df['tokens'] = df[code_key].map(tokenize_fn)
    df = df[df['tokens'].map(lambda x: x is not None and len(x) < limit_length)].copy()
    df = df.drop(columns=['tokens'], axis=1)
    return df


def maintain_function_co_firstlineno(ori_fn):
    """
    This decorator is used to make the decorated function's co_firstlineno the same as the ori_fn
    """

    def wrapper(fn):
        wrapper_code = fn.__code__
        fn.__code__ = types.CodeType(
            wrapper_code.co_argcount,
            wrapper_code.co_kwonlyargcount,
            wrapper_code.co_nlocals,
            wrapper_code.co_stacksize,
            wrapper_code.co_flags,
            wrapper_code.co_code,
            wrapper_code.co_consts,
            wrapper_code.co_names,
            wrapper_code.co_varnames,
            wrapper_code.co_filename,
            wrapper_code.co_name,
            ori_fn.__code__.co_firstlineno,
            wrapper_code.co_lnotab,
            wrapper_code.co_freevars,
            wrapper_code.co_cellvars
        )

        return fn

    return wrapper


def show_process_map(fn, l, print_steps=1000, error_default_value=None):
    res = []
    begin_time = time.time()
    fail_number = 0
    for i, t in enumerate(l):
        if i % print_steps == 0:
            print("{}/{} finished".format(i, len(l)))
            print("{}/{} data map failed".format(fail_number, len(l)))
        try:
            res.append(fn(t))
        except Exception as e:
            # print(e)
            fail_number += 1
            res.append(error_default_value)
    print("This map use {} seconds".format(time.time()-begin_time))
    print("{}/{} data map failed".format(fail_number, len(l)))
    return res


def inplace_show_process_map(fn, l, print_steps=1000, error_default_value=None):
    begin_time = time.time()
    fail_number = 0
    for i, t in enumerate(l):
        if i % print_steps == 0:
            print("{}/{} finished".format(i, len(l)))
            print("{}/{} data map failed".format(fail_number, len(l)))
        try:
            res = fn(t)
        except Exception as e:
            # print(e)
            fail_number += 1
            res = error_default_value
        l[i] = res
    print("This map use {} seconds".format(time.time() - begin_time))
    print("{}/{} data map failed".format(fail_number, len(l)))
    return l


@toolz.curry
def generate_mask(mask_index, size):
    '''
    :param mask_index: a iterable container of index
    :param size: the max size
    :return: a 0-1 mask list with the size shape
    '''
    res = MaskList(size, 0)
    for i in mask_index:
        if isinstance(i, int):
            res.set_mask(i)
        elif isinstance(i, tuple):
            res.set_mask(i[0], i[1])
    res.sort()
    return res


def _get_dataset_item(x):
    return x[0][x[1]]


def data_loader(dataset, batch_size, is_shuffle=True, drop_last=False, epoch_ratio=1.0, multi_process=False):
    idxs = list(range(len(dataset)))
    if is_shuffle:
        idxs = shuffle(idxs)
    idxs = idxs[0: int(len(idxs)*epoch_ratio)]
    # print("the idxs length:{}".format(len(idxs)))
    for idx in batch_holder(idxs, batch_size=batch_size)():
        idx = idx[0]
        if drop_last and len(idx) != batch_size:
            # print("drop_last:{}".format(drop_last))
            # print("len(idx) != batch_size: {}".format(len(idx) != batch_size))
            # print("to break")
            break
        batch = [dataset[i] for i in idx]
        # print("before yield")
        yield toolz.merge_with(lambda x:x, batch)


_end_object = "<END>"


def _inner_queue_data_loader(queue, dataset, batch_size, is_shuffle=True, drop_last=False, epoch_ratio=1.0):
    t = 0
    for batch_data in data_loader(dataset, batch_size, is_shuffle=is_shuffle, drop_last=drop_last, epoch_ratio=epoch_ratio):
        queue.put(batch_data)
        print("{}th pre parsed".format(t))
        t+=1
    queue.put(_end_object)


def queued_data_loader(dataset, batch_size, is_shuffle=True, drop_last=False, epoch_ratio=1.0,
                       queue_size=20):
    import multiprocessing as mp
    from multiprocessing import Queue
    from multiprocessing import Process
    q = Queue(maxsize=queue_size)
    p = mp.Process(target=_inner_queue_data_loader, args=(q, dataset, batch_size, is_shuffle,
                                                          drop_last, epoch_ratio))
    p.start()
    i = 0
    while True:
        t = q.get()
        print("get {}th batch data".format(i))
        i+=1
        if t == _end_object:
            break
        else:
            yield t

# ---------------------------------- PaddedList ------------------------------------------- #

class PaddedList(collections.Sequence):
    """
    list() -> new empty list
    list(iterable) -> new list initialized from iterable's items
    """

    def __init__(self, l, fill_value=0, shape=None):
        self.l = l
        self.fill_value = fill_value

        self.shape = self._l_shape(self.l) if shape is None else shape


    def _l_shape(self, l):
        if not isinstance(l, collections.Sized) and not isinstance(l, collections.Iterable):
            return []
        sh = [len(l)]

        cur_max_shape = None
        for one in l:
            one_shape = self._l_shape(one)
            cur_max_shape = self._cal_max_shapes(cur_max_shape, one_shape) if cur_max_shape is not None else one_shape

        if cur_max_shape is not None:
            sh += cur_max_shape
        return sh

    def _cal_max_shapes(self, ori_shape, one_shape):
        if len(ori_shape) != len(one_shape):
            raise ShapeDifferentException('Shape error. There are different shape in list. original shape is {}, current shape is {}'.format(ori_shape, one_shape))

        max_shape = []
        for ori, one in zip(ori_shape, one_shape):
            max_shape += [max(ori, one)]
        return max_shape

    # make sure the len(l_shape) == len(shape). This example l = [1, 2, 3], shape = [4, 4] will not appear.
    # the fill list and fill number will always append to the end
    def _create_list_as_shape(self, l, shape, fill_value=0):
        if not isinstance(l, collections.Sized) and not isinstance(l, collections.Iterable):
            if len(shape) > 0:
                raise ListShapeErrorException('the depth of list is smaller than len(shape).')
        if len(shape) <= 0:
            raise ListShapeErrorException('shape <= 0')
        # fill value to list
        if len(shape) == 1:
            tmp = [fill_value for i in range(shape[0] - len(l))]
            t = l + tmp
            return t
        # Recursive call _create_list_as_shape
        res = []
        for i in l:
            one = self._create_list_as_shape(i, shape[1:])
            res += [one]
        # add fill list
        if len(l) < shape[0]:
            for i in range(shape[0] - len(l)):
                res += [self._create_list_as_shape([], shape[1:])]
        elif len(l) > shape[0]:
            raise ListShapeErrorException('dim of list is larger than shape. l_len: {}, shape: {}'.format(len(l), shape[0]))
        return res

    def to_list(self):
        res = self._create_list_as_shape(self.l, self.shape, self.fill_value)
        return res

    def __getitem__(self, item):
        ori = item
        if isinstance(item, int):
            if item < 0:
                item += len(self)
            if item < 0 or item > len(self):
                raise IndexError('The index {} is out of range {}'.format(ori, len(self)))
            if len(self.shape) == 1:
                res = self.l[item] if item < len(self.l) else self.fill_value
                return res
            if item >= len(self.l) and item < self.shape[0]:
                return PaddedList([], fill_value=self.fill_value, shape=self.shape[1:])
            elif item >= self.shape[0]:
                raise IndexError('index out of range. list length: {}, i: {}'.format(self.shape[0], item))
            return PaddedList(self.l[item], fill_value=self.fill_value, shape=self.shape[1:])
        elif isinstance(item, slice):
            # len(self.l) == shape[0] should be True. In other word, the first dim should be full.
            tmp_sli = [self.l[ii] for ii in range(*item.indices(len(self)))]
            tmp_shape = [len(tmp_sli)] + self.shape[1:]
            return PaddedList(tmp_sli, fill_value=self.fill_value, shape=tmp_shape)
        else:
            raise TypeError('Invalid argument type. except int or slice but fount {}'.format(type(item)))

    def __len__(self):
        return self.shape[0]

    def __contains__(self, item):
        for i in self:
            if i == item:
                return True
        return False

    def __iter__(self):
        if len(self.shape) == 1:
            for i in range(len(self.l)):
                yield self.l[i]
            for i in range(len(self.l), self.shape[0]):
                yield self.fill_value
        else:
            for i in range(len(self.l)):
                yield PaddedList(self.l[i], fill_value=self.fill_value, shape=self.shape[1:])
            for i in range(len(self.l), self.shape[0]):
                yield PaddedList([], fill_value=self.fill_value, shape=self.shape[1:])

    def __reversed__(self):
        l_len = len(self.l)
        if len(self.shape) == 1:
            for i in range(l_len, self.shape[0]):
                yield self.fill_value
            for i in range(l_len):
                yield self.l[l_len - i - 1]
        else:
            for i in range(l_len, self.shape[0]):
                yield PaddedList([], fill_value=self.fill_value, shape=self.shape[1:])
            for i in range(l_len):
                yield PaddedList(self.l[l_len - i - 1], fill_value=self.fill_value, shape=self.shape[1:])

    def __eq__(self, other):
        if isinstance(other, PaddedList):
            if other.l == self.l and other.shape == self.shape and other.fill_value == self.fill_value:
                return True
        return False

    def __ne__(self, other):
        if isinstance(other, PaddedList):
            if other.l == self.l and other.shape == self.shape and other.fill_value == self.fill_value:
                return False
        return True

    def index(self, x, start: int = ..., end: int = ...):
        for i in range(len(self)):
            if self[i] == x:
                return i
        return -1

    def count(self, x):
        cou = 0
        for i in self:
            if i == x:
                cou += 1
        return cou


class ShapeDifferentException(Exception):
    pass


class ListShapeErrorException(Exception):
    pass


def key_transform(transform, *key, ):
    def transform_fn(sample):
        if len(key) == 1:
            sample[key[0]] = transform(sample[key[0]])
        else:
            in_sample = {k: sample[k] for k in key}
            res = transform(in_sample)
            for k in key:
                del sample[k]
            sample = {**sample, **res}

        # print("sample:{}".format(sample))
        return sample

    return transform_fn


class CopyMap(object):
    def __call__(self, sample):
        return copy.copy(sample)


class IsNone(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, sample):
        if sample is None:
            print("{} is None".format(self._name))
        return sample


class FlatMap(object):
    """
    This map the sample dict to a flat map
    """
    def __call__(self, sample: dict):
        res = {}

        def add_(d: dict):
            for k, v in d.items():
                if not isinstance(v, dict):
                    res[k] = v
                else:
                    add_(v)
        add_(sample)
        return res


def index_select(seq: typing.List, index: typing.List[int]):
    return [seq[k] for k in index]


def filter_token_ids(token_ids, start, end, unk):

    def filter_special_token(token_ids, val):
        return list(filter(lambda x: x != val, token_ids))

    try:
        end_position = token_ids.index(end)
        token_ids = token_ids[:end_position]
    except ValueError as e:
        end_position = None
    # token_ids = filter_special_token(token_ids, start)
    # token_ids = filter_special_token(token_ids, end)
    token_ids = filter_special_token(token_ids, unk)
    return token_ids, end_position

def convert_one_token_ids_to_code(token_ids, id_to_word_fn, start, end, unk, includes=None):
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    token_ids, end_pos = filter_token_ids(token_ids, start, end, unk)
    tokens = [id_to_word_fn(tok) for tok in token_ids]
    code = ' '.join(tokens)
    if includes is not None:
        for inc in includes:
            code = (inc + '\n') + code
    return code, end_pos


def compile_syntax_c_code_by_gcc(code, file_path):
    file_path = add_pid_to_file_path(file_path)
    # target_file_path = add_pid_to_file_path(target_file_path)
    write_code_to_file(code, file_path)
    res = os.system('gcc -fsyntax-only -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    # res = os.system('gcc -c -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    if res == 0:
        return True
    return False


def compile_c_code_by_gcc(code, file_path, target_file_path='main.out', add_pid=True, log_file_path=None):
    if add_pid:
        file_path = add_pid_to_file_path(file_path)
        target_file_path = add_pid_to_file_path(target_file_path)
    write_code_to_file(code, file_path)
    # print(code)
    if log_file_path is None:
        # res = os.system('gcc -o {} -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(target_file_path, file_path))
        res = os.system('gcc -o {} -std=gnu99 {} >/dev/null 2>/dev/null'.format(target_file_path, file_path))
        # res = os.system('gcc -o {} -std=gnu99 {} > nul 2> nul'.format(target_file_path, file_path))
    else:
        log_file_path = add_pid_to_file_path(log_file_path)
        # res = os.system('gcc -o {} -pedantic-errors -std=gnu99 {} >{} 2>&1'.format(target_file_path, file_path, log_file_path))
        res = os.system('gcc -o {} -std=gnu99 {} >{} 2>&1'.format(target_file_path, file_path, log_file_path))
    # res = os.system('gcc -o {} -pedantic-errors -std=gnu99 {}'.format(target_file_path, file_path))
    # res = os.system('gcc -o {} -pedantic-errors -std=gnu99 {} > nul 2> nul'.format(target_file_path, file_path))
    if res == 0:
        return True
    return False


def compile_c_code_by_gcc_c89(code, file_path):
    file_path = add_pid_to_file_path(file_path)
    # target_file_path = add_pid_to_file_path(target_file_path)
    write_code_to_file(code, file_path)
    res = os.system('gcc -pedantic-errors -std=gnu89 {} >/dev/null 2>/dev/null'.format(file_path))
    # res = os.system('gcc -pedantic-errors -std=gnu89 {}'.format(file_path))
    if res == 0:
        return True
    return False


def compile_cpp_code_by_gcc(code, file_path):
    file_path = add_pid_to_file_path(file_path)
    # target_file_path = add_pid_to_file_path(target_file_path)
    write_code_to_file(code, file_path)
    # res = os.system('g++ -c -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    res = os.system('g++ {} >/dev/null 2>/dev/null'.format(file_path))
    # res = os.system('g++ {}'.format(file_path))
    # print('g++ -I/usr/local/include -std=gnu++98 {}'.format(file_path))
    if res == 0:
        return True
    return False


def tokenize_cpp_code_by_new_tokenize(code, print_exception=False):
    try:
        if code.find('define') != -1 or code.find('defined') != -1 or code.find('undef') != -1 or \
                        code.find('pragma') != -1 or code.find('ifndef') != -1 or \
                        code.find('ifdef') != -1 or code.find('endif') != -1:
            return None
        tokens = tokenize(code)
        if len(tokens) > 2000:
            return None
        return tokens
    except Exception as e:
        if print_exception:
            print(e)
        return None


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

class MaskList(collections.Sequence):
    def __init__(self, length, default_value):
        self._length = length
        self._default_value = default_value
        self._label_value = 1 - default_value
        self._mask_segment = []

    def __getitem__(self, i: int):
        for a, b in self._mask_segment:
            if a <= i <= b:
                return 1 - self._default_value
        return self._default_value

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        cur_pos = 0
        for a, b in self._mask_segment:
            if cur_pos < a:
                for i in range(a-cur_pos):
                    yield self._default_value
            for i in range(b-a+1):
                yield self._label_value
            cur_pos = b+1
        if cur_pos < self._length:
            for i in range(self._length-cur_pos):
                yield self._default_value

    def set_mask(self, begin, end=None):
        if end is None:
            end = begin
        self._mask_segment.append((begin, end))

    def flip(self):
        self._default_value = 1 - self._default_value
        self._label_value = 1 - self._default_value
        return self

    def sort(self):
        self._mask_segment = sorted(self._mask_segment, key=lambda x: x[0])
        return self

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True


def transform_id_to_token(one_sequence_ids, id_to_word_fn, offset=0):
    # if not isinstance(one_sequence_ids[0], int):
    #     one_sequence_ids = [i.item() for i in one_sequence_ids]
    # if isinstance(one_sequence_ids, int):
    #     pass
    # else:
    tokens = [id_to_word_fn(i+offset) for i in one_sequence_ids]
    return tokens



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def weight_choice(weight):
    """
    :param weight: list对应的权重序列
    :return:选取的值在原列表里的索引
    """
    t = random.uniform(0, sum(weight))
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i


# ----------------- init source code read from codeforce database directly ------------------------------ #

def check_ascii_character(code:str):
    return all(ord(c) < 128 for c in code)


def init_code(code):
    code = code.replace('\ufeff', '').replace('\u3000', ' ')
    code = remove_blank(code)
    code = remove_r_char(code)
    code = remove_comments(code)
    code = remove_blank_line(code)
    return code


def remove_comments(code):
    pattern = r"(\".*?(?<!\\)\"|\'.*?(?<!\\)\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, code)


def remove_blank_line(code):
    code = "\n".join([line for line in code.split('\n') if line.strip() != ''])
    return code


def remove_r_char(code):
    code = code.replace('\r', '')
    return code


def remove_blank(code):
    pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
    mat = re.findall(pattern, code)
    processed_code = ' '.join(mat)
    return processed_code


def create_token_set(one_ids, vocab):
    total_set = set(one_ids)
    total_set = total_set | vocab.special_token_ids
    return total_set


def create_token_mask_by_token_set(token_set, vocab_len):
    mask = [0 for i in range(vocab_len)]
    for t in token_set:
        mask[t] = 1
    return mask


# ------------------ lex token util method ----------------------- #
def build_code_string_from_lex_tokens(tokens):
    """
    This function build the original code string from the token iterator
    :param tokens: Token iterator
    :return: code string
    """
    lex_tokens = iter(tokens)
    code_re = ""
    lino_pre = 1
    lexpos_pre = 0
    lexpos_temp = 0
    lenth_v = 0
    for token in lex_tokens:
        lino_temp = token.lineno
        if (lino_temp != lino_pre):
            code_re = code_re + "\n"*(lino_temp - lino_pre)
            lenth_v = lino_temp - lino_pre + 1
        else:
            code_re = code_re
        lino_pre = token.lineno
        lexpos_temp = token.lexpos
        code_re = code_re + " " * (lexpos_temp - lexpos_pre - lenth_v)
        code_re = code_re + str(token.value)
        lexpos_pre = lexpos_temp
        lenth_v = len(str(token.value))

    # print(code_re)
    return code_re


def iterate_directory(root_path, extensions=None, recursive=False):
    """
    iterate file in directory
    :param root_path:
    :param extensions:
    :param recursive:
    :return:
    """

    if not recursive:
        for file in os.listdir(root_path):
            file_name, extension_name = os.path.splitext(file)
            if extensions is None\
                    or ((isinstance(extensions, list) or isinstance(extensions, tuple)) and extension_name in extensions) \
                    or (isinstance(extensions, str) and extensions == extension_name):
                yield os.path.join(root_path, file), file
    else:
        for dir_path, dir_names, file_names in os.walk(root_path):
            for file in file_names:
                file_name, extension_name = os.path.splitext(file)
                if extensions is None \
                        or ((isinstance(extensions, list) or isinstance(extensions, tuple)) and extension_name in extensions) \
                        or (isinstance(extensions, str) and extensions == extension_name):
                    yield os.path.join(dir_path, file), file


class CustomerDataSet(Dataset):

    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary,
                 set_type: str,
                 transform=None,
                 no_filter=False):
        super().__init__()
        self.set_type = set_type
        self.transform = transform
        self.vocabulary = vocabulary
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df
            self._samples = [row for i, row in self.data_df.iterrows()]
            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)
            # for s in self._samples:
            #     for k, v in s.items():
            #         print("{}:shape {}".format(k, np.array(v).shape))

    def filter_df(self, df):
        raise NotImplementedError

    def _get_raw_sample(self, row):
        raise NotImplementedError

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [row for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = CustomerDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type, transform=self.transform)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = CustomerDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type, transform=self.transform)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __len__(self):
        return len(self._samples)


class OrderedList(list):
    def __init__(self, s: set):
        super(OrderedList, self).__init__(sorted(s))
        self.stored_set = s

    def __contains__(self, item):
        return item in self.stored_set


def get_position(l, t):
    for i, m in enumerate(l):
        if m == t:
            return i
    return -1


def compile_c_code_by_gcc_one_arg(one):
    return compile_c_code_by_gcc(*one)


def compile_and_read_error_info_one_arg(one):
    return compile_and_read_error_info(*one)


def compile_code_ids_list(final_output, continue_list, result_list, vocabulary, includes_list, file_path='',
                          target_file_path='main.out', log_file_path='main.log', do_compile_pool=True, need_transform=True):
    compile_pool = get_compile_pool()
    batch_size = len(final_output)
    cur_continue = [True for _ in range(batch_size)]
    cur_result_list = [False for _ in range(batch_size)]
    compile_args_list = []
    code_index_dict = []

    count_i = 0
    for code_list, con, includes in zip(final_output, continue_list, includes_list):
        if not con:
            cur_continue[count_i] = False
            res = result_list[count_i]
            cur_result_list[count_i] = res
            count_i += 1
            continue
        if need_transform:
            code_list = [vocabulary.id_to_word(c) for c in code_list]
        code = ' '.join(code_list)
        for inc in includes:
            code = inc + '\n' + code
        compile_args_list += [(code, file_path, target_file_path, log_file_path)]
        code_index_dict += [count_i]
        count_i += 1
    if do_compile_pool:
        # part_res_list = list(compile_pool.starmap(compile_c_code_by_gcc, compile_args_list))
        part_res_list = list(compile_pool.starmap(compile_and_read_error_info, compile_args_list))
    else:
        # part_res_list = map(compile_c_code_by_gcc_one_arg, compile_args_list)
        part_res_list = map(compile_and_read_error_info_one_arg, compile_args_list)

    error_count_list = [-1 for _ in range(batch_size)]
    for i, (res, msg) in enumerate(part_res_list):
        error_list = extract_error_message(msg)
        act_i = code_index_dict[i]
        cur_result_list[act_i] = res
        c = not res
        cur_continue[act_i] = c
        error_count_list[act_i] = len(error_list)

    return cur_continue, cur_result_list, error_count_list


if __name__ == '__main__':
    o = OrderedList({3, 1, 2})
    print(2 in o)
    print(4 in o)
    print(o)


def create_effect_keyword_ids_set(keyword_vocab):
    from common.constants import pre_defined_c_label, pre_defined_c_library_tokens, pre_defined_c_tokens
    # keyword = pre_defined_c_label | pre_defined_c_library_tokens
    keyword = pre_defined_c_tokens | pre_defined_c_library_tokens
    effect_vocabulary_word = keyword_vocab.word_to_id_dict.keys()
    keyword_ids = [keyword_vocab.word_to_id(key) if key in effect_vocabulary_word else None for key in keyword]
    keyword_ids = set(filter(lambda x: x is not None, keyword_ids))
    return keyword_ids


def add_pid_to_file_path(file_path):
    file_name, ext = os.path.splitext(file_path)
    pid = str(os.getpid())
    file_path = file_name + '_' + pid + ext
    return file_path


def create_random_action(ac_code_list):
    from common.action_constants import ActionType
    random_delete = random.randint(0, len(ac_code_list) - 1)
    actions = [{'act_type': ActionType.DELETE, 'from_char': ac_code_list[random_delete],
                'to_char': '', 'token_pos': random_delete}]
    return actions


def retokenize_error_code(error_code_names_list, tokenize_fn):
    new_error_code_names_list = []
    for error_code_list in error_code_names_list:
        err_code = ' '.join(error_code_list)
        tokens = tokenize_fn(err_code)
        one_error_tokens = [tok.value for tok in tokens]
        new_error_code_names_list += [one_error_tokens]
    return new_error_code_names_list


def save_addition_data(original_states, states, tokenize_fn, batch_size, file_path, target_file_path, vocabulary=None,
                       max_distande=None, only_error=False, save_list=None):
    from common.reinforcement_generate_util import generate_action_between_two_code
    save_data_dict = {'ac_code': [], 'action_character_list': [], 'includes': [],
                      'error_count': [], 'distance': [], 'id': []}

    ac_code_names_list = original_states['input_seq_name']
    error_code_ids_list = [c[1:l - 1] for c, l in zip(states['input_seq'], states['copy_length'])]
    error_code_names_list = states['input_seq_name']

    for ids, c in zip(error_code_ids_list, states['copy_length']):
        for p in ids:
            if p > 5941:
                a = 1

    error_code_names_list = retokenize_error_code(error_code_names_list, tokenize_fn)

    do_compile_check = True
    if do_compile_check:
        compile_list = ac_code_names_list + error_code_names_list
        continue_list = [True for _ in range(len(compile_list))]
        last_res_list = [False for _ in range(len(compile_list))]
        include_list = original_states['includes'] + original_states['includes']

        _, compile_res_list, _ = compile_code_ids_list(compile_list, continue_list, last_res_list,
                                                    vocabulary=vocabulary,
                                                    includes_list=include_list, file_path=file_path,
                                                    target_file_path=target_file_path, do_compile_pool=True,
                                                    need_transform=False)
        ac_res_list = compile_res_list[:len(ac_code_names_list)]
        error_res_list = compile_res_list[len(ac_code_names_list):]
    else:
        ac_res_list = [True for _ in range(len(ac_code_names_list))]
        error_res_list = [False for _ in range(len(ac_code_names_list))]

    pool = get_compile_pool()
    max_distance_list = [None for _ in range(batch_size)]
    generate_args = list(zip(error_code_names_list, ac_code_names_list, max_distance_list))
    generate_result = list(pool.starmap(generate_action_between_two_code, generate_args))
    # generate_result = list(itertools.starmap(generate_action_between_two_code, generate_args))
    distance_list, action_list = list(zip(*generate_result))

    print_save_data = False
    if print_save_data:
        for i in range(batch_size):
            info('--------------------------- in save data {} batch ------------------------------------'.format(i))
            ac_full_code = ' '.join(ac_code_names_list[i])
            error_full_code = ' '.join(error_code_names_list[i])
            actions = action_list[i]
            dis = distance_list[i]
            info('ac_code : {}'.format(ac_full_code))
            info('err_code: {}'.format(error_full_code))
            info('dis: {}'.format(dis))
            info('actions: {}'.format(str(actions)))
            info('effect batch: {}'.format(save_list[i]))

    a = 1

    if save_list is None:
        save_list = [True for _ in range(len(ac_code_names_list))]

    for ac_code_list, inc, prog_id, ac_res, err_res, actions, dis, sav \
            in zip(ac_code_names_list, original_states['includes'], original_states['id'],
                   ac_res_list, error_res_list, action_list, distance_list, save_list):
        if not sav:
            continue

        if dis < 0:
            continue
        if max_distande is not None and dis > max_distande:
            continue

        if only_error and err_res:
            continue

        # if 0 > dis or dis >= max_generate_distance:
        #     continue

        ac_code = ' '.join(ac_code_list)
        if len(actions) == 0:
            actions = create_random_action(ac_code_list)
        save_data_dict['ac_code'] += [ac_code]
        save_data_dict['action_character_list'] += [actions]
        save_data_dict['includes'] += [inc]
        save_data_dict['error_count'] += [dis]
        save_data_dict['distance'] += [dis]
        save_data_dict['id'] += [prog_id]
    return save_data_dict


def create_special_tokens_ids(keyword_vocab, has_delimiter=False):
    keyword_ids = create_effect_keyword_ids_set(keyword_vocab)
    special_tokens = []
    special_tokens += keyword_vocab.begin_tokens
    special_tokens += keyword_vocab.end_tokens
    special_tokens += [keyword_vocab.unk]
    special_tokens += keyword_vocab.addition_tokens
    if has_delimiter:
        special_tokens += ['<Delimiter>']
    special_ids = set([keyword_vocab.word_to_id(t) for t in special_tokens])

    total_ids = keyword_ids | special_ids
    return total_ids


def create_special_token_mask_list(total_ids, vocabulary_size):
    mask = [1 if i in total_ids else 0 for i in range(vocabulary_size)]
    return mask


def compile_and_read_error_info(code, file_path='/dev/shm/main.c', target_file_path='/dev/shm/main.out',
                                log_file_path='/dev/shm/main.log'):
    # file_path = '/dev/shm/main.c'
    # target_file_path = '/dev/shm/main.out'
    # log_file_path = '/dev/shm/main.log'
    res = compile_c_code_by_gcc(code, file_path=file_path, target_file_path=target_file_path,
                                log_file_path=log_file_path, add_pid=True)
    pid_log_file_path = add_pid_to_file_path(log_file_path)
    with open(pid_log_file_path, encoding='utf-8') as f:
        texts = f.read()
        texts = texts.replace(u"\u2018", "'").replace(u"\u2019", "'")
    return res, texts


def extract_error_lines(l):
    pattern = re.compile(r"^(.+\.c:)?\d+:\d+: error: (.*)$")
    match = pattern.search(l)
    if match:
        message = match.group(2)
        return message
    return None


def extract_error_message(info):
    info_lines = info.split('\n')
    info_res = [extract_error_lines(l) for l in info_lines]
    error_l = list(filter(lambda x: x is not None, info_res))
    return error_l