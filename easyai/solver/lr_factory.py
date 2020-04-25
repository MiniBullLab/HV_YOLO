#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

REGISTERED_LR_SCHEDULER_CLASSES = {}


def register_lr_scheduler(cls, name=None):
    global REGISTERED_LR_SCHEDULER_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_LR_SCHEDULER_CLASSES, \
        "exist class: {}".format(REGISTERED_LR_SCHEDULER_CLASSES)
    REGISTERED_LR_SCHEDULER_CLASSES[name] = cls
    return cls


def get_lr_scheduler_class(name):
    global REGISTERED_LR_SCHEDULER_CLASSES
    assert name in REGISTERED_LR_SCHEDULER_CLASSES, \
        "available class: {}".format(REGISTERED_LR_SCHEDULER_CLASSES)
    return REGISTERED_LR_SCHEDULER_CLASSES[name]


class LrSchedulerFactory():

    def __init__(self, base_lr, max_epochs=0, epoch_iteration=0):
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.epoch_iteration = epoch_iteration
        self.total_iters = max_epochs * epoch_iteration

    def get_lr_scheduler(self, config):
        lr_class_name = config['lr_type'].strip()
        warm_epoch = config.get('warm_epoch', -1)
        warmup_iters = config.get('warmup_iters', 2000)
        lr_secheduler_cls = get_lr_scheduler_class(lr_class_name)
        result = None
        if lr_class_name == "LinearIncreaseLR":
            end_lr = config['end_lr']
            result = lr_secheduler_cls(self.base_lr, end_lr, self.total_iters,
                                       warm_epoch, warmup_iters)
        elif lr_class_name == "MultiStageLR":
            lr_stages = config['lr_stages']
            assert type(lr_stages) in [list, tuple] and \
                   len(lr_stages[0]) == 2 and lr_stages[-1][0] == self.max_epochs, \
                'lr_stages must be list or tuple, with [iters, lr] format'
            result = lr_secheduler_cls(self.base_lr, lr_stages,
                                       warm_epoch, warmup_iters)
        elif lr_class_name == "PolyLR":
            lr_power = config.get('lr_power', 0.9)
            result = lr_secheduler_cls(self.base_lr, self.total_iters, lr_power,
                                       warm_epoch, warmup_iters)
        elif lr_class_name == "CosineLR":
            result = lr_secheduler_cls(self.base_lr, self.total_iters,
                                       warm_epoch, warmup_iters)
        else:
            print("%s not exit" % lr_class_name)
        return result
