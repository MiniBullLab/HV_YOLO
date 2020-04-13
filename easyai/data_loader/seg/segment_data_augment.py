#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import random
import numpy as np
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class SegmentDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.dataset_process = ImageDataSetProcess()
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        target = label[:]
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, target = self.augment_affine(image, target)
        if self.is_lr_flip:
            image, target = self.augment_lr_flip(image, target)
        return image, target

    def augment_affine(self, src_image, label):
        image_size = (src_image.shape[1], src_image.shape[0])
        matrix, degree = self.dataset_process.affine_matrix(image_size,
                                                            degrees=(-5, 5),
                                                            translate=(0.1, 0.1),
                                                            scale=(0.8, 1.1),
                                                            shear=(-3, 3))
        image = self.dataset_process.image_affine(src_image, matrix,
                                                  border_value=(127.5, 127.5, 127.5))
        target = self.dataset_process.image_affine(label, matrix,
                                                   border_value=250)
        return image, target

    def augment_lr_flip(self, src_image, label):
        image = src_image[:]
        target = label[:]
        if random.random() > 0.5:
            image = np.fliplr(image)
            target = np.fliplr(target)
        return image, target
