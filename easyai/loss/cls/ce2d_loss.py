#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


class CrossEntropy2d(BaseLoss):
    def __init__(self, weight_type=0, weight=None,
                 reduce=None, reduction='mean', ignore_index=250):
        super().__init__(LossType.CrossEntropy2d)
        self.weight_type = weight_type
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index
        if weight_type == 0:
            self.loss_function = nn.CrossEntropyLoss(weight=self.weight,
                                                     reduce=self.reduce,
                                                     reduction=self.reduction,
                                                     ignore_index=self.ignore_index)
        else:
            self.loss_function = nn.CrossEntropyLoss(reduce=self.reduce,
                                                     reduction=self.reduction,
                                                     ignore_index=self.ignore_index)

    def compute_loss_from_weight(self, loss, target):
        weight = [float(x) for x in self.weight.split(',') if x]
        return loss

    def forward(self, outputs, target=None):
        outputs = outputs.float()
        if target is not None:
            loss = self.loss_function(outputs, target)
            if self.weight_type != 0 and self.weight is not None:
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = F.softmax(outputs, dim=1)
        return loss

