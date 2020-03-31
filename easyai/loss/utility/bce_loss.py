#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


class BinaryCrossEntropy2d(BaseLoss):

    def __init__(self, weight_type=0, weight=None,
                 reduce=None, reduction='mean'):
        super().__init__(LossType.BinaryCrossEntropy2d)
        self.weight_type = weight_type
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction
        if weight_type == 0:
            self.loss_function = torch.nn.BCELoss(weight=self.weight, reduce=self.reduce,
                                                  reduction=self.reduction)
        else:
            self.loss_function = torch.nn.BCELoss(reduce=self.reduce, reduction=self.reduction)

    def compute_loss_from_weight(self, loss, target):
        weight = [float(x) for x in self.weight.split(',') if x]
        if self.weight_type == 1:
            result = weight[0] * target.le(0.5).type(loss.dtype) * loss + \
                     weight[1] * target.gt(0.5).type(loss.dtype) * loss
        else:
            result = loss
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result

    def forward(self, input_data, target=None):
        if target is not None:
            loss = self.loss_function(input_data, target)
            if self.weight_type != 0 and self.weight is not None:
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = input_data
        return loss
