#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.loss.utility.bce_loss import BinaryCrossEntropy2d


class EncNetLoss(BaseLoss):

    def __init__(self, num_class,
                 se_loss=False, se_weight=0.2,
                 aux=False, aux_weight=0.4,
                 weight=None, reduction='mean', ignore_index=250):
        super().__init__(LossType.EncNetLoss)
        self.num_class = num_class
        self.se_loss = se_loss
        self.aux = aux
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.ce = CrossEntropy2d(0, weight, True,
                                 reduction, ignore_index)
        self.bce = BinaryCrossEntropy2d(0, weight, True,
                                        reduction, ignore_index)

    def forward(self, input_datas, target=None):
        if target is not None:
            if not self.se_loss and not self.aux:
                result = self.ce(input_datas[0], target)
            elif not self.se_loss:
                loss1 = self.ce(input_datas[0], target)
                loss2 = self.ce(input_datas[1], target)
                result = loss1 + self.aux_weight * loss2
            elif not self.aux:
                se_target = self._get_batch_label_vector(target, self.num_class).type_as(input_datas[0])
                loss1 = self.ce(input_datas[0], target)
                loss2 = self.bceloss(torch.sigmoid(input_datas[2]), se_target)
                result = loss1 + self.se_weight * loss2
            else:
                se_target = self._get_batch_label_vector(target, self.num_class).type_as(input_datas[0])
                loss1 = self.ce(input_datas[0], target)
                loss2 = self.ce(input_datas[1], target)
                loss3 = self.bceloss(torch.sigmoid(input_datas[2]), se_target)
                result = loss1 + self.aux_weight * loss2 + self.se_weight * loss3
        else:
            result = self.ce(input_datas[0])
        return result

    @staticmethod
    def _get_batch_label_vector(target, num_class):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, num_class))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=num_class, min=0,
                               max=num_class-1)
            vect = hist > 0
            tvect[i] = vect
        return tvect
