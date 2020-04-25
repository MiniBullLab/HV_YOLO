#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc


class BaseLrSecheduler():
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_lr(self, cur_epoch, cur_iter):
        pass

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def multi_learning_rate(self, optimizer, lr):
        for i in enumerate(optimizer.param_groups):
            if i == 0:
                optimizer.param_groups[i]['lr'] = lr
            else:
                optimizer.param_groups[i]['lr'] = lr * 10
