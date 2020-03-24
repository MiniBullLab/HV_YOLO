#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
from easyai.solver.base_lr_secheduler import BaseLrSecheduler
from easyai.solver.lr_factory import register_lr_scheduler


@register_lr_scheduler
class LinearIncreaseLR(BaseLrSecheduler):
    def __init__(self, base_lr, end_lr, total_iters,
                 warm_epoch=-1, warmup_iters=2000):
        super().__init__()
        self.baseLr = base_lr
        self.endLr = end_lr
        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            return self.endLr + (self.baseLr - self.endLr) * (1 - float(cur_iter) / self.total_iters)


@register_lr_scheduler
class MultiStageLR(BaseLrSecheduler):
    def __init__(self, base_lr, lr_stages,
                 warm_epoch=-1, warmup_iters=2000):
        super().__init__()
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self.baseLr = base_lr
        self.lr_stages_list = lr_stages
        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            for it_lr in self.lr_stages_list:
                if cur_epoch < it_lr[0]:
                    return self.baseLr * it_lr[1]


@register_lr_scheduler
class PolyLR(BaseLrSecheduler):
    def __init__(self, base_lr, total_iters, lr_power=0.9,
                 warm_epoch=-1, warmup_iters=2000):
        super().__init__()
        self.baseLr = base_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            return self.baseLr * ((1 - float(cur_iter) / self.total_iters) ** self.lr_power)


@register_lr_scheduler
class CosineLR(BaseLrSecheduler):
    def __init__(self, base_lr, total_iters,
                 warm_epoch=-1, warmup_iters=5):
        super().__init__()
        self.baseLr = base_lr
        self.total_iters = total_iters + 0.0

        self.warm_epoch = warm_epoch
        self.warmup_iters = warmup_iters

    def get_lr(self, cur_epoch, cur_iter):
        if (cur_epoch == self.warm_epoch) and (cur_iter <= self.warmup_iters):
            lr = self.baseLr * (cur_iter / self.warmup_iters) ** 4
            return lr
        else:
            return self.baseLr * (1 + math.cos(math.pi * float(cur_iter) / self.total_iters)) / 2
