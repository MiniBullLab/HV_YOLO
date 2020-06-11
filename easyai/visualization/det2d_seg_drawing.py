#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
from easyai.visualization.image_drawing import ImageDrawing


class Det2dSegTaskShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, result,
             is_gray, class_name, detection_objects, scale=0.8):
        segment_image = self.drawing.draw_segment_result(src_image, result,
                                                         is_gray, class_name)

        self.drawing.drawDetectObjects(segment_image, detection_objects)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(segment_image.shape[1] * scale), int(segment_image.shape[0] * scale))
        cv2.imshow("image", segment_image)
        if cv2.waitKey() & 0xff == ord('q'):  # 按q退出
            return False
        else:
            return True
