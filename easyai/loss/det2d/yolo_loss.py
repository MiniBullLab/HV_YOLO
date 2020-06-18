#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.det2d_gt_process import Det2dGroundTruthProcess


class YoloLoss(BaseLoss):

    def __init__(self, name, class_number, anchor_sizes=None):
        super().__init__(name)
        self.class_number = class_number
        if anchor_sizes is not None:
            self.anchor_sizes = torch.Tensor(anchor_sizes)
            self.anchor_count = len(anchor_sizes)
            self.anchor_step = len(anchor_sizes[0])
        else:
            self.anchor_sizes = ()
            self.anchor_count = 0
            self.anchor_step = 0
        self.gt_process = Det2dGroundTruthProcess()

    def decode_predict_box(self, coord, N, H, W, device):
        all_count = N * self.anchor_count * H * W
        pred_boxes = torch.zeros(all_count, 4, dtype=torch.float, device=device)
        lin_x = torch.linspace(0, W - 1, W).to(device).repeat(H, 1).view(H * W)
        lin_y = torch.linspace(0, H - 1, H).to(device).repeat(W, 1).t().contiguous().view(H * W)
        anchor_w = self.anchors[:, 0].view(self.anchor_count, 1).to(device)
        anchor_h = self.anchors[:, 1].view(self.anchor_count, 1).to(device)

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        return pred_boxes

    def decode_predict_points(self, x_point, y_point, point_count,
                              N, H, W, device):
        # Create pred points
        all_count = N * 1 * H * W
        temp_count = N * 1
        pred_corners = torch.zeros(point_count * 2, all_count, dtype=torch.float, device=device)
        grid_x = torch.linspace(0, W - 1, W).to(device).repeat(H, 1).\
            repeat(temp_count, 1, 1).view(all_count)
        grid_y = torch.linspace(0, H - 1, H).to(device).repeat(W, 1).t().\
            repeat(temp_count, 1, 1).view(all_count)
        for i in range(0, self.keypoint_count, 2):
            pred_corners[i] = (x_point[i].data.contiguous().view_as(grid_x) + grid_x)
            pred_corners[i + 1] = (y_point[i].data.contiguous().view_as(grid_y) + grid_y)
        return pred_corners

    def scale_anchor(self):
        if self.anchor_step == 4:
            anchors = self.anchor_sizes.clone()
            anchors[:, :2] = 0
        else:
            anchors = torch.cat([torch.zeros_like(self.anchor_sizes), self.anchor_sizes], 1)
        return anchors

    def print_info(self):
        info_str = ''
        for key, value in self.info:
            info_str += "%s: %.5f|" % value
        print('%s' % info_str)
