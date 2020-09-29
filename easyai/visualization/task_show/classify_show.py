#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
from easyai.visualization.utility.image_drawing import ImageDrawing


class ClassifyShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, result,
             class_name, scale=1.0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(src_image, class_name[result],
                    (int(src_image.shape[0] * 0.1), int(src_image.shape[1] * 0.1)),
                    font, 0.5, (0, 255, 0), 2)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(src_image.shape[1] * scale), int(src_image.shape[0] * scale))
        cv2.imshow('image', src_image)

        if cv2.waitKey() & 0xFF == 27:
            return False
        else:
            return True
