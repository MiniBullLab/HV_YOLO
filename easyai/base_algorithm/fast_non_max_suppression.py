#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.base_algorithm.base_non_max_suppression import BaseNonMaxSuppression


class FastNonMaxSuppression(BaseNonMaxSuppression):

    def __init__(self):
        super().__init__()
        self.nms_style = "OR"
        self.sigma = 0.5

    def multi_class_nms(self, input_objects, nms_threshold, score_threshold=0.5):
        result = []
        if len(input_objects) == 0:
            return result

        inputs = self.sort_detect_objects(input_objects, 1)
        class_index_list = np.unique(inputs[:, -1])

        temp_result = []
        for class_index in class_index_list:
            class_objects = inputs[inputs[:, -1] == class_index]
            temp = self.nms(class_objects, nms_threshold, score_threshold)
            temp_result.extend(temp)

        result = self.numpy_to_objects(temp_result)
        return result

    def nms(self, numpy_objects, nms_threshold, score_threshold=0.5):
        result = []
        count = len(numpy_objects)
        if count == 1:
            result.append(numpy_objects)
            return result
        if self.nms_style == 'OR':
            result = self.or_nms(numpy_objects, nms_threshold)
        elif self.nms_style == "AND":
            result = self.and_nms(numpy_objects, nms_threshold)
        elif self.nms_style == 'MERGE':  # weighted mixture box
            result = self.merge_nms(numpy_objects, nms_threshold)
        elif self.nms_style == 'SOFT_Linear':
            result = self.soft_linear_nms(numpy_objects, nms_threshold, score_threshold)
        elif self.nms_style == 'SOFT_Gaussian':
            result = self.soft_gaussian_nms(numpy_objects, nms_threshold, score_threshold)
        else:
            print("nms function error!")
        return result

    def or_nms(self, numpy_objects, nms_threshold):
        result = []
        while numpy_objects.shape[0] > 0:
            result.append(numpy_objects[:1])  # save highest conf detection
            if len(numpy_objects) == 1:  # Stop if we're at the last detection
                break
            ious = self.numpy_box_iou(numpy_objects[0], numpy_objects[1:])  # iou with other boxes
            numpy_objects = numpy_objects[1:][ious < nms_threshold]  # remove ious > threshold
        return result

    def and_nms(self, numpy_objects, nms_threshold):
        result = []
        while numpy_objects.shape[0] > 1:
            ious = self.numpy_box_iou(numpy_objects[0], numpy_objects[1:])  # iou with other boxes
            if np.max(ious) > 0.5:
                result.append(numpy_objects[:1])
            numpy_objects = numpy_objects[1:][ious < nms_threshold]  # remove ious > threshold
        return result

    def merge_nms(self, numpy_objects, nms_threshold):
        result = []
        while numpy_objects.shape[0] > 0:
            ious = self.numpy_box_iou(numpy_objects[0], numpy_objects)  # iou with other boxes
            index_list = ious > nms_threshold  # iou with other boxes
            weights = numpy_objects[index_list, 4:5]
            numpy_objects[0, :4] = np.sum(weights * numpy_objects[index_list, :4], axis=0) / np.sum(weights)
            result.append(numpy_objects[:1])
            numpy_objects = numpy_objects[index_list == 0]
        return result

    def soft_linear_nms(self, numpy_objects, nms_threshold, score_threshold):
        result = []
        while numpy_objects.shape[0] > 0:
            # Get detection with highest confidence and save as max detection
            result.append(numpy_objects[0])
            # Stop if we're at the last detection
            if numpy_objects.shape[0] == 1:
                break
            ious = self.numpy_box_iou(result[-1], numpy_objects[1:])  # iou with other boxes
            weight = (ious > nms_threshold) * (1 - ious) + (ious < nms_threshold)
            numpy_objects[1:, 4] *= weight
            numpy_objects = numpy_objects[1:]
            index_list = numpy_objects[:, 4] > score_threshold
            numpy_objects = numpy_objects[index_list]
            # Stop if we're at the last detection
            if numpy_objects.shape[0] == 0:
                break
            sort_index = np.argsort(numpy_objects[:, 4], axis=0)[::-1]
            numpy_objects = numpy_objects[sort_index]
        return result

    def soft_gaussian_nms(self, numpy_objects, nms_threshold, score_threshold):
        result = []
        while numpy_objects.shape[0] > 0:
            # Get detection with highest confidence and save as max detection
            result.append(numpy_objects[0])
            # Stop if we're at the last detection
            if numpy_objects.shape[0] == 1:
                break
            ious = self.numpy_box_iou(result[-1], numpy_objects[1:])  # iou with other boxes
            # ious = (ious > nms_threshold) * (1 - ious) + (ious < nms_threshold)
            weight = np.exp(- (ious * ious) / self.sigma)
            numpy_objects[1:, 4] *= weight
            numpy_objects = numpy_objects[1:]
            index_list = numpy_objects[:, 4] > score_threshold
            numpy_objects = numpy_objects[index_list]
            # Stop if we're at the last detection
            if numpy_objects.shape[0] == 0:
                break
            sort_index = np.argsort(numpy_objects[:, 4], axis=0)[::-1]
            numpy_objects = numpy_objects[sort_index]
        return result
