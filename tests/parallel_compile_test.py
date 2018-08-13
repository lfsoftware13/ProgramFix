import time

import torch
import torch.multiprocessing as mp

from common.util import compile_c_code_by_gcc, add_pid_to_file_path

code = r'''
#include<stdio.h>

int main(){
    int a=0, b=0;
    printf("a %d, b %d", a, b);
    return 0;
}
'''


def single_compile(t):
    file_path = '/dev/shm/main.c'
    target_file_path = '/dev/shm/main.out'
    file_path = add_pid_to_file_path(file_path)
    target_file_path = add_pid_to_file_path(target_file_path)
    res = compile_c_code_by_gcc(code, file_path, target_file_path)


if __name__ == '__main__':
    total_times = 120
    num_processes = 6
    multi_times = total_times // num_processes
    a = time.time()
    for i in range(120):
        single_compile(1)
    b = time.time()
    print('single compile time: {}'.format(b-a))

    pool = mp.Pool(num_processes)
    c = time.time()
    for i in range(multi_times):
        pool.map(single_compile, [[] for _ in range(num_processes)])
    d = time.time()
    print('parallel compile time: {}'.format(d-c))


