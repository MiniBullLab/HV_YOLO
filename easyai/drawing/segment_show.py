#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
from easyai.drawing.image_drawing import ImageDrawing


class SegmentionShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, result,
             is_gray, class_name, scale=0.5):
        segment_image = self.drawing.draw_segment_result(src_image, result,
                                                         is_gray, class_name)

        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(segment_image.shape[1] * scale), int(segment_image.shape[0] * scale))
        cv2.imshow('image', segment_image)

        if cv2.waitKey() & 0xFF == 27:
            return False
        else:
            return True
