# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 10:58
# @Author  : wlz
# @Project   : TextLevelGNN
# @File    : main.py
# @Software: PyCharm

import time
import random
import numpy as np
import torch
from modules.model import GNNModel
from modules.optimizer import Optimizer
from config.conf import arg_config, path_config
from utils.datautil import create_vocab, batch_variable
import torch.nn.functional as F
from logger.logger import logger
from utils.dataset import DataSet, DataLoader
import torch.nn.utils as nn_utils


class Trainer(object):
    def __init__(self, args, vocabs):
        self.vocabs = vocabs
        self.args = args
        self.model = GNNModel(num_node=len(vocabs['word']),
                              embedding_dim=args.wd_embed_dim,
                              num_cls=len(vocabs['label']),
                              pre_embed=vocabs['word'].embeddings).to(args.device)
        print(self.model)
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def set_dataset(self, data_path):
        train_set = DataSet(data_path['dataset']['train'])
        self.train_set, self.val_set = train_set.split(split_rate=0.1, shuffle=True)
        self.test_set = DataSet(data_path['dataset']['test'])
        print(f'Train Size: {len(self.train_set)}, Val Size: {len(self.val_set)}, Test Size: {len(self.test_set)}')

    def train(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Optimizer(params, args)
        patient = 0
        best_dev_acc, best_test_acc = 0, 0
        for ep in range(1, self.args.epoch+1):
            train_loss, train_acc = self.train_iter(ep, self.train_set, optimizer)

            dev_acc = self.eval(self.val_set)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                test_acc = self.eval(self.test_set)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                patient = 0
            else:
                patient += 1

            logger.info('[Epoch %d] train loss: %.4f, lr: %f, Train ACC: %.4f, Dev ACC: %.4f, Best Dev ACC: %.4f, Best Test ACC: %.4f, patient: %d' % (
                    ep, train_loss, optimizer.get_lr(), train_acc, dev_acc, best_dev_acc, best_test_acc, patient))

            if patient >= args.patient:
                break

        logger.info('Final Test ACC: %.4f' % best_test_acc)

    def train_iter(self, ep, train_set, optimizer):
        t1 = time.time()
        train_acc, train_loss = 0., 0.
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        self.model.train()
        for i, batcher in enumerate(train_loader):
            batch = batch_variable(batcher, self.vocabs)
            batch.to_device(self.args.device)
            pred = self.model(batch.x, batch.nx, batch.ew)
            loss = F.nll_loss(pred, batch.y)
            loss.backward()
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=args.grad_clip)
            optimizer.step()
            self.model.zero_grad()

            loss_val = loss.data.item()
            train_loss += loss_val
            train_acc += (pred.data.argmax(dim=-1) == batch.y).sum().item()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train acc: %.4f, train loss: %.4f' % (
                ep, i + 1, (time.time() - t1), optimizer.get_lr(), train_acc/len(train_set), loss_val))

        return train_loss/len(train_set), train_acc/len(train_set)

    def eval(self, test_set):
        nb_correct, nb_total = 0, 0
        test_loader = DataLoader(test_set, batch_size=self.args.test_batch_size)
        self.model.eval()
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                pred = self.model(batch.x, batch.nx, batch.ew)
                nb_correct += (pred.data.argmax(dim=-1) == batch.y).sum().item()
                nb_total += len(batch.y)
        return nb_correct / nb_total


if __name__ == '__main__':
    np.random.seed(2343)
    random.seed(1347)
    torch.manual_seed(1453)
    torch.cuda.manual_seed(1347)
    torch.cuda.manual_seed_all(1453)

    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = arg_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    data_path = path_config('./config/data_path.json')
    vocabs = create_vocab(data_path['dataset']['train'])
    embed_count = vocabs['word'].load_embeddings(data_path['pre_embed']['word_embedding'])
    print("%d pre-trained embeddings loaded..." % embed_count)

    trainer = Trainer(args, vocabs)
    trainer.set_dataset(data_path)
    trainer.train()
