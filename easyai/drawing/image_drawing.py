#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
import numpy as np
from easyai.drawing.colorDefine import ColorDefine


class ImageDrawing():

    def __init__(self):
        pass

    def drawDetectObjects(self, srcImage, detectObjects):
        for object in detectObjects:
            point1 = (int(object.min_corner.x), int(object.min_corner.y))
            point2 = (int(object.max_corner.x), int(object.max_corner.y))
            index = object.classIndex
            cv2.rectangle(srcImage, point1, point2, ColorDefine.colors[index], 2)

    def draw_segment_result(self, src_image, result,
                            is_gray, class_list):
        r = result.copy()
        g = result.copy()
        b = result.copy()
        for index, value_data in enumerate(class_list):
            value = value_data[1]
            if is_gray:
                gray_value = int(value.strip())
                r[result == index] = gray_value
                g[result == index] = gray_value
                b[result == index] = gray_value
            else:
                color_list = [int(x) for x in value.spilt(',') if x.strip()]
                r[result == index] = color_list[0]
                g[result == index] = color_list[1]
                b[result == index] = color_list[2]

        rgb = np.zeros((result.shape[0], result.shape[1], 3))

        rgb[:, :, 0] = (r * 1.0 + src_image[:, :, 2] * 0) / 255.0
        rgb[:, :, 1] = (g * 1.0 + src_image[:, :, 1] * 0) / 255.0
        rgb[:, :, 2] = (b * 1.0 + src_image[:, :, 0] * 0) / 255.0

        return rgb
