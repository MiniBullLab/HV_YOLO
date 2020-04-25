#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


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
