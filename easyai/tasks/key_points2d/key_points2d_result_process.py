#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import torch
import numpy as np
from easyai.helper.dataType import Point2d, DetectionObject
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class KeyPoints2dResultProcess():

    def __init__(self, points_count):
        self.use_new_confidence = False
        self.points_count = points_count
        self.dataset_process = ImageDataSetProcess()

    def get_keypoints_result(self, prediction, conf_thresh, flag=0):
        result = None
        if flag == 0:
            result = self.get_yolo_result(prediction, conf_thresh)
        elif flag == 1:
           pass
        return result

    def get_yolo_result(self, prediction, conf_thresh):
        result = []
        loc_count = self.points_count * 2
        class_confidence, class_index = prediction[:, loc_count + 1:].max(1)
        if self.use_new_confidence:
            object_confidence = prediction[:, loc_count]
            object_confidence *= class_confidence
            temp1_indexs = object_confidence > conf_thresh
        else:
            temp1_indexs = prediction[:, loc_count] > conf_thresh
        temp2_indexs = torch.isfinite(prediction).all(1)
        index_list = temp1_indexs & temp2_indexs
        prediction = prediction[index_list]
        class_confidence = class_confidence[index_list]
        class_index = class_index[index_list]
        # (x1, y1, x2, y2, ...)
        best_confidence = -1
        for index, value in enumerate(prediction):
            temp_object = DetectionObject()
            for index in range(0, self.points_count, 2):
                point = Point2d(value[index], value[index+1])
                temp_object.add_key_points(point)
            temp_object.objectConfidence = value[4]
            temp_object.classConfidence = class_confidence[index]
            temp_object.classIndex = class_index[index]
            if best_confidence == -1:
                result.append(temp_object)
                best_confidence = temp_object.objectConfidence
            elif temp_object.objectConfidence > best_confidence:
                result[0] = temp_object
        return result

    def resize_keypoints_objects(self, src_size, image_size,
                                 result_objects, class_name):
        result = []
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for obj in result_objects:
            temp_object = DetectionObject()
            for index in range(0, self.points_count, 2):
                temp_object.key_points[index].x = (temp_object.key_points[index].x -
                                                   pad[0] // 2) / ratio
                temp_object.key_points[index].y = (temp_object.key_points[index].y -
                                                   pad[1] // 2) / ratio
            temp_object.classIndex = int(obj.classIndex)
            temp_object.objectConfidence = obj.objectConfidence
            temp_object.classConfidence = obj.classConfidence
            temp_object.name = class_name[temp_object.classIndex]
            result.append(temp_object)
        return result

