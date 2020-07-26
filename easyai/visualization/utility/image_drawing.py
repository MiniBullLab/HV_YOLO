#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
import numpy as np
from easyai.visualization.utility.color_define import ColorDefine


class ImageDrawing():

    def __init__(self):
        pass

    def drawDetectObjects(self, src_image, result_objects):
        for object in result_objects:
            point1 = (int(object.min_corner.x), int(object.min_corner.y))
            point2 = (int(object.max_corner.x), int(object.max_corner.y))
            index = object.classIndex
            cv2.rectangle(src_image, point1, point2, ColorDefine.colors[index], 2)

    def draw_segment_result(self, src_image, result, class_list):
        r = result.copy()
        g = result.copy()
        b = result.copy()
        for index, value_data in enumerate(class_list):
            value = value_data[1]
            color_list = [int(x) for x in value.split(',') if x.strip()]
            if len(color_list) == 1:
                r[result == index] = color_list[0]
                g[result == index] = color_list[0]
                b[result == index] = color_list[0]
            elif len(color_list) == 3:
                r[result == index] = color_list[0]
                g[result == index] = color_list[1]
                b[result == index] = color_list[2]
            else:
                r[result == index] = 0
                g[result == index] = 0
                b[result == index] = 0

        rgb = np.zeros((result.shape[0], result.shape[1], 3))

        rgb[:, :, 0] = (r * 1.0 + src_image[:, :, 2] * 0) / 255.0
        rgb[:, :, 1] = (g * 1.0 + src_image[:, :, 1] * 0) / 255.0
        rgb[:, :, 2] = (b * 1.0 + src_image[:, :, 0] * 0) / 255.0

        return rgb

    def draw_keypoints_result(self, src_image, result):
        edges_corners = [[1, 2], [2, 4], [4, 3], [3, 1], [1, 5], [5, 6],
                         [6, 8], [8, 7], [7, 5], [7, 3], [8, 4], [6, 2]]
        for result_object in result:
            key_points = result_object.get_key_points()
            for edge in edges_corners:
                point1 = key_points[edge[0]]
                point2 = key_points[edge[1]]
                cv2.line(src_image, (point1.x, point1.y), (point2.x, point2.y), (0, 0, 255), 2)
