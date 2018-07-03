

GPU_INDEX = 0


def to_cuda(x):
    if GPU_INDEX is not None:
        x = x.cuda(GPU_INDEX)
    return x
