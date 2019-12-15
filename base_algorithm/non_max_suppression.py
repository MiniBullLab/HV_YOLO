#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class NonMaxSuppression():

    def __init__(self):
        self.nms_style = "OR"
        self.is_GIoU = False

    def multi_class_nms(self, input_objects, threshold):
        result = []
        class_index_list = self.get_class_name(input_objects)
        sort_inputs = self.numpy_sort_objects(input_objects)
        for class_index in class_index_list:
            class_objects = [input_objects[index] for index in sort_inputs
                             if input_objects[index].classIndex == class_index]
            temp_result = self.nms(class_objects, threshold)
            result.extend(temp_result)
        return result

    def nms(self, input_objects, threshold):
        result = []
        if self.nms_style == 'OR':
            result = self.or_nms(input_objects, threshold)
        elif self.nms_style == 'AND':  # requires overlap, single boxes erased
            result = self.and_nms(input_objects, threshold)
        elif self.nms_style == 'MERGE':  # weighted mixture box
            pass
        elif self.nms_style == 'SOFT_Linear':
            pass
        elif self.nms_style == 'SOFT_Gaussian':
            pass
        return result

    def or_nms(self, input_objects, threshold):
        result = []
        count = len(input_objects)
        remove_indexs = [False for _ in range(count)]
        for loop1 in range(count):
            if not remove_indexs[loop1]:
                result.append(input_objects[loop1])
            else:
                continue
            for loop2 in range(loop1+1, count):
                if not remove_indexs[loop2]:
                    iou = self.compute_object_iou(input_objects[loop1],
                                                  input_objects[loop2])
                    if iou >= threshold:
                        remove_indexs[loop2] = True
        return result

    def and_nms(self, input_objects, threshold):
        result = []
        count = len(input_objects)
        if count == 1:
            return input_objects
        remove_indexs = [False for _ in range(count)]
        for loop1 in range(count):
            if remove_indexs[loop1]:
                continue
            max_iou = 0
            for loop2 in range(loop1 + 1, count):
                if not remove_indexs[loop2]:
                    iou = self.compute_object_iou(input_objects[loop1],
                                                  input_objects[loop2])
                    max_iou = max(max_iou, iou)
                    if iou >= threshold:
                        remove_indexs[loop2] = True
            if max_iou > 0.5:
                result.append(input_objects[loop1])
        return result

    def merge_nms(self, input_objects, threshold):
        result = []
        count = len(input_objects)
        if count == 1:
            return input_objects
        remove_indexs = [False for _ in range(count)]
        for loop1 in range(count):
            if remove_indexs[loop1]:
                continue
            sum_iou = 0
            sum_x1 = 0
            sum_y1 = 0
            sum_x2 = 0
            sum_y2 = 0
            for loop2 in range(count):
                if not remove_indexs[loop2]:
                    iou = self.compute_object_iou(input_objects[loop1],
                                                  input_objects[loop2])
                    if iou >= threshold:
                        sum_x1 += iou * input_objects[loop2].min_corner.x
                        sum_y1 += iou * input_objects[loop2].min_corner.y
                        sum_x2 += iou * input_objects[loop2].max_corner.x
                        sum_y2 += iou * input_objects[loop2].max_corner.y
                        sum_iou += iou
                        remove_indexs[loop2] = True
            if sum_iou > 0:
                input_objects[loop1].min_corner.x = sum_x1 / sum_iou
                input_objects[loop1].min_corner.y = sum_y1 / sum_iou
                input_objects[loop1].max_corner.x = sum_x2 / sum_iou
                input_objects[loop1].max_corner.y = sum_y2 / sum_iou
                result.append(input_objects[loop1])
        return result

    def soft_linear(self, input_objects, threshold):
        pass

    def soft_gaussian(self, input_objects, threshold):
        pass

    def get_class_name(self, input_objects):
        temp_value = [x.classIndex for x in input_objects]
        return set(temp_value)

    def numpy_sort_objects(self, input_objects):
        temp_value = [x.objectConfidence for x in input_objects]
        temp_value = np.array(temp_value)
        result = np.argsort(temp_value, axis=0)
        return result

    def sort_objects(self, input_objects):
        result = sorted(input_objects, key=lambda x: x.objectConfidence,
                        reverse=True)
        return result

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

