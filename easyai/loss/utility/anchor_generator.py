#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
import torch
import numpy as np


class SSDPriorBoxGenerator():

    def __init__(self):
        self.anchor_size = None
        self.aspect_ratio_list = None
        self.anchor_count = None
        self.clip = False

    def set_anchor_param(self, anchor_count, anchor_size, aspect_ratio_list):
        self.anchor_count = anchor_count
        self.anchor_size = anchor_size
        self.aspect_ratio_list = aspect_ratio_list

    def __call__(self, feature_size, input_size):
        image_w, image_h = input_size
        feature_map_w, feature_map_h = feature_size
        stride_w = image_w / feature_map_w
        stride_h = image_h / feature_map_h

        boxes = []
        stride_offset_w, stride_offset_h = 0.5 * stride_w, 0.5 * stride_h
        s_w = self.anchor_size[0]
        s_h = self.anchor_size[0]
        boxes.append((stride_offset_w, stride_offset_h, s_w, s_h))
        extra_s = math.sqrt(self.anchor_size[0] * self.anchor_size[1])
        boxes.append((stride_offset_w, stride_offset_h, extra_s, extra_s))

        for ratio in self.aspect_ratio_list:
            boxes.append((stride_offset_w, stride_offset_h,
                          s_w * math.sqrt(ratio), s_h / math.sqrt(ratio)))
            boxes.append((stride_offset_w, stride_offset_h,
                          s_w / math.sqrt(ratio), s_h * math.sqrt(ratio)))

        anchor_bases = torch.FloatTensor(np.array(boxes))
        assert anchor_bases.size(0) == self.anchor_count
        anchors = anchor_bases.contiguous().view(1, -1, 4).\
            repeat(feature_map_h * feature_map_w, 1, 1).contiguous().view(-1, 4)
        grid_len_h = np.arange(0, image_h - stride_offset_h, stride_h)
        grid_len_w = np.arange(0, image_w - stride_offset_w, stride_w)
        a, b = np.meshgrid(grid_len_w, grid_len_h)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
        x_y_offset = x_y_offset.repeat(1, self.anchor_count, 1).contiguous().view(-1, 2)
        anchors[:, :2] = anchors[:, :2] + x_y_offset

        if self.clip:
            anchors[:, 0::2].clamp_(min=0., max=image_w - 1)
            anchors[:, 1::2].clamp_(min=0., max=image_h - 1)

        return anchors
