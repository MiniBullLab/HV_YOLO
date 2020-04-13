#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
from easyai.drawing.image_drawing import ImageDrawing


class DetectionShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, detection_objects):
        self.drawing.drawDetectObjects(src_image, detection_objects)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(src_image.shape[1] * 0.8), int(src_image.shape[0] * 0.8))
        cv2.imshow("image", src_image)
        if cv2.waitKey() & 0xff == ord('q'):  # 按q退出
            return False
        else:
            return True
