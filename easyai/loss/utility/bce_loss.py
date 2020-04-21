#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *
from easyai.loss.utility.calculate_weights import numpy_compute_weight


class BinaryCrossEntropy2d(BaseLoss):

    def __init__(self, weight_type=0, weight=None,
                 reduce=None, reduction='mean', ignore_index=250):
        super().__init__(LossType.BinaryCrossEntropy2d)
        self.weight_type = weight_type
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index
        if weight_type == 0:
            self.loss_function = torch.nn.BCELoss(weight=self.weight, reduce=self.reduce,
                                                  reduction=self.reduction)
        else:
            self.loss_function = torch.nn.BCELoss(reduce=self.reduce, reduction=self.reduction)

    def compute_loss_from_weight(self, loss, target):
        if self.weight_type == 1:
            weights = [float(x) for x in self.weight.split(',') if x]
            result = weights[0] * target.eq(0).type(loss.dtype) * loss + \
                     weights[1] * target.eq(1).type(loss.dtype) * loss
        elif self.weight_type == 2:
            labels = target.data.cpu().numpy()
            weights = numpy_compute_weight(labels)
            result = weights[0] * target.eq(0).type(loss.dtype) * loss + \
                     weights[1] * target.eq(1).type(loss.dtype) * loss
        else:
            result = loss
        result = target.ne(self.ignore_index).type(loss.dtype) * result
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result

    def forward(self, input_data, target=None):
        if target is not None:
            loss = self.loss_function(input_data, target)
            if self.weight_type != 0:
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = input_data
        return loss
