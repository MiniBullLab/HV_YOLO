#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
import numpy as np
import random


class ImageDataAugment():

    def __init__(self):
        pass

    def augment_hsv(self, rgb_image):
        # SV augmentation by 50%
        fraction = 0.50
        img_hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        a = (random.random() * 2 - 1) * fraction + 1
        S *= a
        if a > 1:
            np.clip(S, a_min=0, a_max=255, out=S)

        a = (random.random() * 2 - 1) * fraction + 1
        V *= a
        if a > 1:
            np.clip(V, a_min=0, a_max=255, out=V)

        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return result
