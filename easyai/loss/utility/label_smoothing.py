#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch


class LabelSmoothing():

    def __init__(self, class_number, epsilon=0.1, ignore_index=-100):
        self.class_number = class_number
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.lb_pos = 1.0 - self.epsilon
        self.lb_neg = self.epsilon / self.class_number

    def smoothing(self, outputs, targets):
        with torch.no_grad():
            labels = targets.clone().detach()
            ignore = labels == self.ignore_index
            labels[ignore] = 0
            smooth_onehot = torch.empty_like(outputs).fill_(
                self.lb_neg).scatter_(1, labels.unsqueeze(1), self.lb_pos).detach()
        return smooth_onehot




