#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.utility.registry import build_from_cfg
from easyai.solver.registry import REGISTERED_LR_SCHEDULER


class LrSchedulerFactory():

    def __init__(self, base_lr, max_epochs=0, epoch_iteration=0):
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.epoch_iteration = epoch_iteration
        self.total_iters = max_epochs * epoch_iteration

    def get_lr_scheduler(self, config):
        self.process_warmup(config)
        lr_class_name = config['type'].strip()
        cfg = config.copy()
        cfg['base_lr'] = self.base_lr
        result = None
        if lr_class_name == "LinearIncreaseLR":
            cfg['total_iters'] = self.total_iters
            result = build_from_cfg(cfg, REGISTERED_LR_SCHEDULER)
        elif lr_class_name == "MultiStageLR":
            result = build_from_cfg(cfg, REGISTERED_LR_SCHEDULER)
        elif lr_class_name == "PolyLR":
            cfg['total_iters'] = self.total_iters
            result = build_from_cfg(cfg, REGISTERED_LR_SCHEDULER)
        elif lr_class_name == "CosineLR":
            cfg['total_iters'] = self.total_iters
            result = build_from_cfg(cfg, REGISTERED_LR_SCHEDULER)
        else:
            print("%s not exit" % lr_class_name)
        return result

    def process_warmup(self, config):
        warmup_type = int(config.get('warmup_type', 0))
        if warmup_type == 2:
            warmup_iters = int(config.get('warmup_iters', 0))
            warmup_iters = warmup_iters * self.epoch_iteration
            config['warmup_iters'] = warmup_iters
