#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


def smooth_l1_loss(x, t):
    diff = (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < 1.).float()
    y = flag * (diff ** 2) * 0.5 + (1 - flag) * (abs_diff - 0.5)
    return y.sum()


class MeanSquaredErrorLoss(BaseLoss):

    def __init__(self, reduction='mean'):
        super().__init__(LossType.MeanSquaredErrorLoss)
        self.loss_function = torch.nn.MSELoss(reduction=reduction)

    def forward(self, input_data, target=None):
        if target is not None:
            loss = self.loss_function(input_data, target)
        else:
            loss = input_data
        return loss

