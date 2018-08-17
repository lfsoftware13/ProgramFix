import torch.nn as nn
import torch.multiprocessing as mp

from config import num_processes

GPU_INDEX = 0
Parallel = False


def get_model(m):
    if Parallel:
        m = nn.DataParallel(m.cuda(), device_ids=[1])
    elif GPU_INDEX is not None:
        m = nn.DataParallel(m.cuda(), device_ids=[GPU_INDEX])
    return m

def to_cuda(x):
    if Parallel:
        return x.cuda()
    elif GPU_INDEX is not None:
        x = x.cuda(GPU_INDEX)
    return x


compile_pool = None
def get_compile_pool():
    global compile_pool
    if compile_pool is None:
        compile_pool = mp.Pool(num_processes)
    return compile_pool

