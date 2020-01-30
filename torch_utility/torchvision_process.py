#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from torchvision.transforms import Compose, CenterCrop
from torchvision.transforms import ToTensor, Resize


class TorchVisionProcess():

    def __init__(self):
        pass

    def input_transform(self, crop_size, upscale_factor):
        return Compose([
            CenterCrop(crop_size),
            Resize(crop_size // upscale_factor),
            ToTensor(),
        ])

    def target_transform(self, crop_size):
        return Compose([
            CenterCrop(crop_size),
            ToTensor(),
        ])

    def calculate_valid_crop_size(self, crop_size, upscale_factor):
        return crop_size - (crop_size % upscale_factor)