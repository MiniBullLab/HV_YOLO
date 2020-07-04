#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch


class Det2dGroundTruthProcess():

    def __init__(self):
        pass

    def scale_gt_box(self, gt_data, width, height):
        gt_box = torch.zeros(len(gt_data), 4)
        for i, anno in enumerate(gt_data):
            gt_box[i, 0] = anno[1] * width
            gt_box[i, 1] = anno[2] * height
            gt_box[i, 2] = anno[3] * width
            gt_box[i, 3] = anno[4] * height
        return gt_box

    def scale_gt_points(self, gt_data, point_count, width, height):
        gx = []
        gy = []
        gt_box = []
        for index in range(1, point_count, 2):
            gt_box.extend([gt_data[index], gt_data[index+1]])
            gx.append(gt_data[index] * width)
            gy.append(gt_data[index+1] * height)
        gt_corners = torch.FloatTensor(gt_box)
        return gt_corners, gx, gy
