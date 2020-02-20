#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.base_algorithm.base_non_max_suppression import BaseNonMaxSuppression


class FastNonMaxSuppression(BaseNonMaxSuppression):

    def __init__(self):
        super().__init__()
        self.nms_style = "OR"

    def multi_class_nms(self, input_objects, threshold):
        result = []
        if len(input_objects) == 0:
            return result

        inputs = self.sort_detect_objects(input_objects, 1)
        class_index_list = inputs[:, -1].unique()

        temp_result = []
        for class_index in class_index_list:
            class_objects = inputs[inputs[:, -1] == class_index]
            temp = self.nms(class_objects, threshold)
            temp_result.extend(temp)

        result = self.numpy_to_objects(temp_result)
        return result

    def nms(self, numpy_objects, threshold):
        result = []
        count = len(numpy_objects)
        if count == 1:
            result.append(numpy_objects)
            return result
        if self.nms_style == 'OR':
            while numpy_objects.shape[0]:
                result.append(numpy_objects[:1])  # save highest conf detection
                if len(numpy_objects) == 1:  # Stop if we're at the last detection
                    break
                iou = self.bbox_iou(numpy_objects[0], numpy_objects[1:])  # iou with other boxes
                numpy_objects = numpy_objects[1:][iou < threshold]  # remove ious > threshold
        return result
