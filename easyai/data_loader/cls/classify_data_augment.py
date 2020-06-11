#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.torch_utility.torch_vision.torchvision_process import TorchVisionProcess
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class ClassifyDataAugment():

    def __init__(self):
        torchvision_process = TorchVisionProcess()
        self.augment_transform = torchvision_process.torch_data_augment()
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
                image, _, _ = self.image_augment.augment_affine(image)
            if self.is_lr_flip:
                image, _ = self.image_augment.augment_lr_flip(image)
        return image
