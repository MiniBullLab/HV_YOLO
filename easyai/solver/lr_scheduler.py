#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
from easyai.solver.base_lr_secheduler import BaseLrSecheduler
from easyai.solver.registry import REGISTERED_LR_SCHEDULER


@REGISTERED_LR_SCHEDULER.register_module
class LinearIncreaseLR(BaseLrSecheduler):
    def __init__(self, base_lr, end_lr, total_iters,
                 warmup_type=0, warmup_iters=2000):
        super().__init__(base_lr, warmup_type, warmup_iters)
        self.endLr = end_lr
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        if self.warmup_type != 0 and (cur_iter <= self.warmup_iters):
            lr = self.get_warmup_lr(cur_iter)
            return lr
        else:
            return self.endLr + (self.baseLr - self.endLr) * (1 - float(cur_iter) / self.total_iters)


@REGISTERED_LR_SCHEDULER.register_module
class MultiStageLR(BaseLrSecheduler):
    def __init__(self, base_lr, lr_stages,
                 warmup_type=0, warmup_iters=2000):
        super().__init__(base_lr, warmup_type, warmup_iters)
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self.lr_stages_list = lr_stages

    def get_lr(self, cur_epoch, cur_iter):
        if self.warmup_type != 0 and (cur_iter <= self.warmup_iters):
            lr = self.get_warmup_lr(cur_iter)
            return lr
        else:
            for it_lr in self.lr_stages_list:
                if cur_epoch < it_lr[0]:
                    return self.baseLr * it_lr[1]


@REGISTERED_LR_SCHEDULER.register_module
class PolyLR(BaseLrSecheduler):
    def __init__(self, base_lr, total_iters, lr_power=0.9,
                 warmup_type=0, warmup_iters=2000):
        super().__init__(base_lr, warmup_type, warmup_iters)
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        if self.warmup_type != 0 and (cur_iter <= self.warmup_iters):
            lr = self.get_warmup_lr(cur_iter)
            return lr
        else:
            return self.baseLr * ((1 - float(cur_iter) / self.total_iters) ** self.lr_power)


@REGISTERED_LR_SCHEDULER.register_module
class CosineLR(BaseLrSecheduler):
    def __init__(self, base_lr, total_iters,
                 warmup_type=0, warmup_iters=5):
        super().__init__(base_lr, warmup_type, warmup_iters)
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_epoch, cur_iter):
        if self.warmup_type != 0 and (cur_iter <= self.warmup_iters):
            lr = self.get_warmup_lr(cur_iter)
            return lr
        else:
            return self.baseLr * (1 + math.cos(math.pi * float(cur_iter) / self.total_iters)) / 2
