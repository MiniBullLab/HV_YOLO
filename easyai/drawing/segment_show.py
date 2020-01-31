#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
from easyai.drawing.image_drawing import ImageDrawing


class SegmentionShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, result, class_name):
        decoded = self.drawing.drawSegmentResult(src_image, result,
                                                 class_name)

        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(decoded.shape[1] * 0.5), int(decoded.shape[0] * 0.5))
        cv2.imshow('image', decoded)

        if cv2.waitKey() & 0xFF == 27:
            return False
        else:
            return True
