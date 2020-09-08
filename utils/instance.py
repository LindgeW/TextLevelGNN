# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 9:51
# @Author  : wlz
# @Project   : TextLevelGNN
# @File    : instance.py
# @Software: PyCharm


class Instance(object):
    def __init__(self, data=None, label=None):
        self.data = data
        self.label = label

    def __repr__(self):
        return f'data: {self.data}, label: {self.label}'

    def __str__(self):
        return f'data: {self.data}, label: {self.label}'
