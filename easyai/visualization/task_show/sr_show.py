#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2


class SuperResolutionShow():

    def __init__(self):
        pass

    def show(self, src_image, sr_image, scale=0.5):
        cv2.namedWindow("src_image", 0)
        cv2.resizeWindow("src_image", int(src_image.shape[1] * scale), int(src_image.shape[0] * scale))
        cv2.imshow('src_image', src_image)

        cv2.namedWindow("sr_image", 0)
        cv2.resizeWindow("sr_image", int(sr_image.shape[1] * scale), int(sr_image.shape[0] * scale))
        cv2.imshow('sr_image', sr_image)

        if cv2.waitKey() & 0xFF == 27:
            return False
        else:
            return True