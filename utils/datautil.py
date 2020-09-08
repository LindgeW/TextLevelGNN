# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 9:41
# @Author  : wlz
# @Project   : TextLevelGNN
# @File    : datautil.py
# @Software: PyCharm

import os
import torch
import numpy as np
from utils.vocab import Vocab, MultiVocab


def create_vocab(data_path):
    wd_vocab = Vocab(min_count=3, bos=None, eos=None)
    lbl_vocab = Vocab(pad=None, unk=None, bos=None, eos=None)
    assert os.path.exists(data_path)
    with open(data_path, 'r', encoding='utf-8') as fin:
        loader = map(lambda x: x.strip().split('|||'), fin)
        for lbl, data_item in loader:
            wds = data_item.strip().split(' ')
            wd_vocab.add(wds)
            lbl_vocab.add(lbl.strip())
    return MultiVocab({'word': wd_vocab, 'label': lbl_vocab})


def batch_variable(insts, vocabs):
    bs = len(insts)
    nb_nodes = len(vocabs['word']) - 1
    max_len = max(len(inst.data) for inst in insts)
    x_ids = torch.zeros((bs, max_len), dtype=torch.long)      # 当前词
    nx_ids = torch.zeros((bs, max_len, 6), dtype=torch.long)  # 临近词
    ew_ids = torch.zeros((bs, max_len, 6), dtype=torch.long)  # 边索引
    y_ids = torch.zeros((bs, ), dtype=torch.long)

    for i, inst in enumerate(insts):
        length = len(inst.data)
        x = np.asarray(vocabs['word'].inst2idx(inst.data))
        y = vocabs['label'].inst2idx(inst.label)
        nx = get_neighbors(x, 3)
        # edge weight index
        ew_id = ((x - 1) * nb_nodes).reshape(-1, 1) + nx
        ew_id[x == 0] = 0
        ew_id[nx == 0] = 0

        x_ids[i, :length] = torch.tensor(x)
        nx_ids[i, :length] = torch.tensor(nx)
        ew_ids[i, :length] = torch.tensor(ew_id)
        y_ids[i] = torch.tensor(y)
    return Batch(x_ids, nx_ids, ew_ids, y_ids)


def get_neighbors(x_ids, nb_neighbor=2):
    neighbours = []
    pad = [0] * nb_neighbor
    x_ids_ = pad + list(x_ids) + pad
    for i in range(nb_neighbor, len(x_ids_) - nb_neighbor):
        x = x_ids_[i-nb_neighbor: i] + x_ids_[i+1: i+nb_neighbor+1]
        neighbours.append(x)
    return np.asarray(neighbours)


class Batch(object):
    def __init__(self, x_tensor, nx_tensor, ew_tensor, y_tensor):
        self.x = x_tensor
        self.nx = nx_tensor
        self.ew = ew_tensor
        self.y = y_tensor

    def to_device(self, device):
        self.x = self.x.to(device)
        self.nx = self.nx.to(device)
        self.ew = self.ew.to(device)
        self.y = self.y.to(device)
        return self
