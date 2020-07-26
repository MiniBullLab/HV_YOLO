#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.loss.utility.base_loss import *
import numpy as np


class OhemCrossEntropy2d(BaseLoss):

    def __init__(self, ignore_index=-1, threshold=0.7, min_kept=int(32 // 1 * 640 * 352 // 16)):
        super().__init__(LossType.OhemCrossEntropy2d)
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_kept = min_kept
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def build_target(self, input_data, target):
        n, c, h, w = input_data.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(input_data, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.threshold
            if self.min_kept > 0:
                # index = mask_prob.argsort()
                index = np.argsort(mask_prob.cpu().detach().numpy())
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return target

    def forward(self, input_data, target):
        if target is not None:
            target = self.compute_ohem_loss(input_data, target)
            loss = self.loss_function(input_data, target)
        else:
            loss = F.softmax(input_data, dim=1)
        return loss