# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 9:41
# @Author  : wlz
# @Project   : TextLevelGNN
# @File    : dataset.py
# @Software: PyCharm
# Dataset: https://github.com/yao8839836/text_gcn/tree/master/data

import os
import numpy as np
from utils.instance import Instance


def load_data(data_path):
    assert os.path.isfile(data_path) and os.path.exists(data_path)
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as fin:
        loader = map(lambda x: x.strip().split('|||'), fin)
        for label, data_item in loader:
            dataset.append(Instance(data_item.strip().split(' '),
                                    label.strip()))
    return dataset


class DataSet(object):
    def __init__(self, insts, transform=None):
        if isinstance(insts, str):
            self.insts = load_data(insts)
        else:
            self.insts = insts
        self.transform = transform

    def from_file(self, file):
        self.insts = load_data(file)

    def from_data(self, insts):
        self.insts = insts

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        sample = self.insts[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        for inst in self.insts:
            yield inst

    def index(self, item):
        return self.insts.index(item)

    def split(self, split_rate=0.33, shuffle=False):
        n = len(self.insts)
        assert self.insts and n > 0
        if shuffle:
            # np.random.shuffle(self.insts)
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)
        val_size = int(n * split_rate)
        # train_set = DataSet(self.insts[:-val_size])
        # val_set = DataSet(self.insts[-val_size:])
        train_set = DataSet([self.insts[idxs[i]] for i in range(val_size, n)])
        val_set = DataSet([self.insts[idxs[i]] for i in range(val_size)])
        return train_set, val_set


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)

        batch = []
        for idx in idxs:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                if self.collate_fn:  # 如: 对齐和tensor化
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        if len(batch) > 0:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


