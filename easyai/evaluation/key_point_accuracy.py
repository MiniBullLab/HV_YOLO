#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class KeyPointAccuracy():

    def __init__(self, class_names):
        self.class_names = class_names

    def eval(self, numpy_outputs, numpy_targets):
        for index, gt_datas in enumerate(numpy_targets):
            output_datas = numpy_outputs[index]
            for class_index in range(len(self.class_names)):
                pr_index = output_datas[:, -1] == class_index
                gt_index = gt_datas[:, -1] == class_index
                pr_points = output_datas[pr_index, :18]
                gt_points = output_datas[gt_index, :18]
                corner_norm = np.linalg.norm(pr_points - gt_points, axis=1)
                corner_dist = np.mean(corner_norm)