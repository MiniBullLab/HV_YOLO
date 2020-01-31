#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.base_loss import *


class BinaryCrossEntropy2d(BaseLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(LossType.BinaryCrossEntropy2d)
        self.weight = weight
        self.reduction = reduction
        self.loss_function = torch.nn.BCELoss(reduce=False)

    def segment_resize(self, input, target):
        target = target.type(input.dtype)
        if input.size() != target.size():
            # remove dim equal to 1
            input_ = input.squeeze()
            target_ = target.squeeze()
            if input_.size() != target.size():
                assert 'Shape must be equal: input size {} != target size'.format(input.size(), target.size())

            return input_, target_
        else:
            return input, target

    def forward(self, input, target=None):
        if target is not None:
            loss = self.loss_function(input, target)
            if self.weight is not None:
                loss = self.weight[0] * target.le(0.5).type(loss.dtype) * loss + \
                       self.weight[1] * target.gt(0.5).type(loss.dtype) * loss
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            loss = input
        return loss
