#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
from easyai.visualization.utility.image_drawing import ImageDrawing


class KeyPointsShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, result_objects):
        self.drawing.draw_keypoints_result(src_image, result_objects)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(src_image.shape[1] * 0.8), int(src_image.shape[0] * 0.8))
        cv2.imshow("image", src_image)
        if cv2.waitKey() & 0xff == ord('q'):
            return False
        else:
            return True
