#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import random
import numpy as np
from easyai.torch_utility.torch_vision.torchvision_process import TorchVisionProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class ClassifyDataAugment():

    def __init__(self):
        torchvision_process = TorchVisionProcess()
        self.augment_transform = torchvision_process.torch_data_augment()
        self.dataset_process = ImageDataSetProcess()
        self.image_augment = ImageDataAugment()
        self.is_torchvision_augment = True
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True

    def augment(self, image_rgb):
        image = image_rgb[:]
        if self.is_torchvision_augment:
            image = self.augment_transform(image)
        else:
            if self.is_augment_hsv:
                image = self.image_augment.augment_hsv(image)
            if self.is_augment_affine:
                image = self.augment_affine(image)
            if self.is_lr_flip:
                image = self.augment_lr_flip(image)
        return image

    def augment_affine(self, src_image):
        image_size = (src_image.shape[1], src_image.shape[0])
        matrix, degree = self.dataset_process.affine_matrix(image_size,
                                                            degrees=(-15, 15),
                                                            translate=(0.0, 0.0),
                                                            scale=(1.0, 1.0),
                                                            shear=(-3, 3))
        image = self.dataset_process.image_affine(src_image, matrix,
                                                  border_value=(0.0, 0.0, 0.0))
        return image

    def augment_lr_flip(self, src_image):
        image = src_image[:]
        if random.random() > 0.5:
            image = np.fliplr(image)
        return image
