import torch.nn as nn

GPU_INDEX = 1
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
