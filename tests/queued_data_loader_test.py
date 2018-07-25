
from common.util import queued_data_loader
from torch.utils.data import Dataset

import numpy as np


class DataSet(Dataset):
    def __init__(self):
        super().__init__()
        self._raw_data = [{"value": [i]*10} for i in range(1000)]

    def __len__(self):
        return len(self._raw_data)

    def __getitem__(self, index):
        return self._raw_data[index]


if __name__ == '__main__':
    dataset = DataSet()
    print("the dataset created")
    for t in queued_data_loader(dataset, 32, is_shuffle=False, drop_last=True):
        print(t)
