#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class KeyPoints2dDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, labels):
        image = image_rgb[:]
        targets = labels[:]
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        return image, targets
