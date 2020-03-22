#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper.dataType import DetectionObject


class BaseNonMaxSuppression():

    def __init__(self):
        self.is_GIoU = False

    def objects_to_numpy(self, input_objects):
        result = np.zeros((len(input_objects), 7), dtype=np.float32)
        for index, temp_object in enumerate(input_objects):
            result[index, :] = np.array([temp_object.min_corner.x,
                                         temp_object.min_corner.y,
                                         temp_object.max_corner.x,
                                         temp_object.max_corner.y,
                                         temp_object.objectConfidence,
                                         temp_object.classConfidence,
                                         temp_object.classIndex])
        return result

    def numpy_to_objects(self, input_numpy):
        result = []
        for value in input_numpy:
            value = np.squeeze(value)
            temp_object = DetectionObject()
            temp_object.min_corner.x = value[0]
            temp_object.min_corner.y = value[1]
            temp_object.max_corner.x = value[2]
            temp_object.max_corner.y = value[3]
            temp_object.objectConfidence = value[4]
            temp_object.classConfidence = value[5]
            temp_object.classIndex = value[6]
            result.append(temp_object)
        return result

    def sort_detect_objects(self, input_objects, flag=0):
        if flag == 0:
            temp_value = [-x.objectConfidence for x in input_objects]
            temp_value = np.array(temp_value)
            indexs = np.argsort(temp_value, axis=0)
            result = indexs
        elif flag == 1:
            inputs = self.objects_to_numpy(input_objects)
            indexs = np.argsort((-inputs[:, 4]))
            result = inputs[indexs]
        else:
            result = sorted(input_objects, key=lambda x: x.objectConfidence,
                            reverse=True)
        return result

    def numpy_box_iou(self, box1, box2):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T

        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0, :], box2[1, :], box2[2, :], box2[3, :]

        min_x = (b2_x1 < b1_x1) * b1_x1 + (b2_x1 > b1_x1) * b2_x1
        min_y = (b2_y1 < b1_y1) * b1_y1 + (b2_y1 > b1_y1) * b2_y1
        max_x = (b2_x2 < b1_x2) * b2_x2 + (b2_x2 > b1_x2) * b1_x2
        max_y = (b2_y2 < b1_y2) * b2_y2 + (b2_y2 > b1_y2) * b1_y2

        # Intersection area
        inter_area = np.maximum(0, max_x - min_x) * np.maximum(0, max_y - min_y)

        # Union Area
        union_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + \
                     (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

        iou = inter_area / (union_area + + 1e-16)  # iou
        if self.is_GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_x1 = (b2_x1 < b1_x1) * b2_x1 + (b2_x1 > b1_x1) * b1_x1
            c_y1 = (b2_y1 < b1_y1) * b2_y1 + (b2_y1 > b1_y1) * b1_y1
            c_x2 = (b2_x2 < b1_x2) * b1_x2 + (b2_x2 > b1_x2) * b2_x2
            c_y2 = (b2_y2 < b1_y2) * b1_y2 + (b2_y2 > b1_y2) * b2_y2
            c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        return iou

    def compute_object_iou(self, object1, object2):
        min_x = max(object1.min_corner.x, object2.min_corner.x)
        min_y = max(object1.min_corner.y, object2.min_corner.y)
        max_x = min(object1.max_corner.x, object2.max_corner.x)
        max_y = min(object1.max_corner.y, object2.max_corner.y)
        width = max(max_x - min_x, 0)
        height = max(max_y - min_y, 0)
        # Intersection area
        inter_area = width * height
        # Union Area
        union_area = object1.width() * object1.height() + \
                     object2.width() * object2.height() - inter_area
        iou = 0
        if union_area > 0:
            iou = inter_area / union_area
            # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            if self.is_GIoU:
                min_x = min(object1.min_corner.x, object2.min_corner.x)
                min_y = min(object1.min_corner.y, object2.min_corner.y)
                max_x = max(object1.max_corner.x, object2.max_corner.x)
                max_y = max(object1.max_corner.y, object2.max_corner.y)
                # convex area
                convex_area = (max_x - min_x) * (max_y - min_y)
                iou = iou - (convex_area - union_area) / convex_area  # GIoU
        return iou
