#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *
from easyai.loss.utility.label_smoothing import LabelSmoothing


class LabelSmoothCE2dLoss(BaseLoss):

    def __init__(self, class_number, epsilon=0.1, reduction='mean', ignore_index=250):
        super().__init__(LossType.LabelSmoothCE2dLoss)
        self.class_number = class_number
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, outputs, targets=None):
        outputs = outputs.float()
        if targets is None:
            loss = F.softmax(outputs, dim=1)
        else:
            log_preds = F.log_softmax(outputs, dim=1)
            if self.reduction == 'sum':
                log_loss = -log_preds.sum()
            else:
                log_loss = -log_preds.sum(dim=-1)
                if self.reduction == 'mean':
                    log_loss = log_loss.mean()
            nll_loss = F.nll_loss(log_preds, targets, reduction=self.reduction,
                                  ignore_index=self.ignore_index)
            loss = log_loss * self.epsilon / self.class_number + (1 - self.epsilon) * nll_loss
        return loss


class LabelSmoothCE2dLossV2(BaseLoss):

    def __init__(self, class_number, epsilon=0.1, reduction='mean', ignore_index=250):
        super().__init__(LossType.LabelSmoothCE2dLoss)
        self.class_number = class_number
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.label_smoothing = LabelSmoothing(class_number, epsilon, ignore_index)

    def forward(self, outputs, targets=None):
        outputs = outputs.float()
        if targets is None:
            loss = F.softmax(outputs, dim=1)
        else:
            ignore = targets == self.ignore_index
            valid_count = (ignore == 0).sum()
            smooth_onehot = self.label_smoothing.smoothing(outputs, targets)
            logs = self.log_softmax(outputs)
            loss = -torch.sum(logs * smooth_onehot, dim=1)
            loss[ignore] = 0
            if self.reduction == 'mean':
                loss = loss.sum() / valid_count
            if self.reduction == 'sum':
                loss = loss.sum()
        return loss
