#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


class PointNetLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossType.PointNetLoss)

    def feature_transform_reguliarzer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss


class PointNetClsLoss(PointNetLoss):

    def __init__(self, flag):
        super().__init__()
        self.flag = flag
        self.mat_diff_loss_scale = 0.001

    def forward(self, input_list, target=None):
        if not isinstance(input_list, list):
            input_x = [input_list]
        else:
            input_x = input_list
        x = F.log_softmax(input_x[0], dim=-1)
        if target is not None:
            loss = F.nll_loss(x, target.long())
            if self.flag:
                mat_diff_loss = self.feature_transform_reguliarzer(input_x[2])
                loss += mat_diff_loss * self.mat_diff_loss_scale
        else:
            loss = x
        return loss


class PointNetSegLoss(PointNetLoss):

    def __init__(self, flag):
        super().__init__()
        self.flag = flag
        self.mat_diff_loss_scale = 0.001

    def forward(self, input_list, target=None):
        if not isinstance(input_list, list):
            input_x = [input_list]
        else:
            input_x = input_list
        x = input_x[0]
        batch_size = x.size()[0]
        num_class = x.size()[2]
        n_pts = x.size()[1]
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, num_class), dim=-1)
        x = x.view(batch_size, n_pts, num_class)
        if target is not None:
            loss = F.nll_loss(x, target.long())
            if self.flag:
                mat_diff_loss = self.feature_transform_reguliarzer(input_x[2])
                loss += mat_diff_loss * self.mat_diff_loss_scale
        else:
            loss = x
        return loss
