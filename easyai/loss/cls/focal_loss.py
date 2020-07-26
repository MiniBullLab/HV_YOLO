#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.loss.utility.base_loss import *


class FocalLoss(BaseLoss):

    def __init__(self, class_number, alpha=0.25, gamma=2,
                 reduction='mean', ignore_index=250):
        super().__init__(LossType.FocalLoss)
        self.class_number = class_number
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, outputs, targets=None):
        outputs = outputs.float()
        if targets is None:
            loss = F.softmax(outputs, dim=1)
        else:
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)  # N,C,H,W => N,C,H*W
                outputs = outputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
                outputs = outputs.contiguous().view(-1, self.class_number)  # N,H*W,C => N*H*W,C
            targets = targets.view(-1, 1)
            ignore = targets == self.ignore_index
            logpt = F.log_softmax(outputs)
            logpt = logpt.gather(1, targets)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())
            focus_p = torch.pow(1 - pt, self.gamma)
            loss = -1 * self.alpha * focus_p * logpt
            loss[ignore] = 0
            if self.reduction == 'mean':
                loss = loss.mean()
            if self.reduction == 'sum':
                loss = loss.sum()
        return loss


class FocalBinaryLoss(BaseLoss):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__(LossType.FocalBinaryLoss)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, outputs, targets=None):
        outputs = outputs.float()
        if targets is None:
            loss = F.softmax(outputs, dim=1)
        else:
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)  # N,C,H,W => N,C,H*W
                outputs = outputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
                outputs = outputs.contiguous().view(-1, 1)  # N,H*W,C => N*H*W,C
            targets = targets.view(-1, 1)
            with torch.no_grad():
                alpha = torch.empty_like(outputs).fill_(1 - self.alpha)
                alpha[targets == 1] = self.alpha
            focus_p = torch.pow(torch.abs(targets - outputs), self.gamma)
            # pt = torch.where(targets == 1, 1 - outputs, outputs)
            # focus_p = torch.pow(1 - pt, self.gamma)
            bce_loss = self.bce(outputs, targets)
            loss = self.alpha * focus_p * bce_loss
            if self.reduction == 'mean':
                loss = loss.mean()
            if self.reduction == 'sum':
                loss = loss.sum()
        return loss
