#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class SegmentDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.label_border_value = 250
        self.dataset_process = ImageDataSetProcess()
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        target = label[:]
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix, _ = self.image_augment.augment_affine(image)
            target = self.label_affine(label, matrix)
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            target = self.label_lr_flip(target, is_lr)
        return image, target

    def label_affine(self, label, matrix):
        target = self.dataset_process.image_affine(label, matrix,
                                                   border_value=self.label_border_value)
        return target

    def label_lr_flip(self, label, is_lr):
        target = label[:]
        if is_lr:
            target = np.fliplr(target)
        return target

    def label_up_flip(self, label, is_up):
        target = label[:]
        if is_up:
            target = np.flipud(target)
        return target
