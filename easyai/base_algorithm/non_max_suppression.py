#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper.dataType import DetectionObject


class NonMaxSuppression():

    def __init__(self):
        self.nms_style = "OR"
        self.is_GIoU = False

    def multi_class_nms(self, input_objects, threshold):
        result = []
        if len(input_objects) == 0:
            return result
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

    def fast_multi_class_nms(self, input_objects, threshold):
        result = []
        if len(input_objects) == 0:
            return result

        inputs = self.objects_to_numpy(input_objects)
        indexs = np.argsort((-inputs[:, 4]))
        inputs = inputs[indexs]
        class_index_list = inputs[:, -1].unique()

        temp_result = []
        for class_index in class_index_list:
            class_objects = inputs[inputs[:, -1] == class_index]
            temp = self.fast_nms(class_objects, threshold)
            temp_result.extend(temp)

        for value in temp_result:
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

    def fast_nms(self, numpy_objects, threshold):
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
                iou = self.bbox_iou(numpy_objects[0], dc[1:])  # iou with other boxes
                dc = dc[1:][iou < threshold]  # remove ious > threshold
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

    def bbox_iou(self, box1, box2):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.t()

        # Get the coordinates of bounding boxes
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.min(b1_x2, b2_x2) - np.max(b1_x1, b2_x1)).clamp(0) * \
                     (np.min(b1_y2, b2_y2) - np.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                     (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

        iou = inter_area / union_area  # iou
        if self.is_GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_x1, c_x2 = np.min(b1_x1, b2_x1), np.max(b1_x2, b2_x2)
            c_y1, c_y2 = np.min(b1_y1, b2_y1), np.max(b1_y2, b2_y2)
            c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        return iou

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


