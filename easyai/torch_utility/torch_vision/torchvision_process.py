#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torchvision.transforms as transforms
from torchvision.transforms import Compose, CenterCrop
from torchvision.transforms import ToTensor, Resize


class TorchVisionProcess():

    def __init__(self):
        pass

    def torch_normalize(self, flag=0, mean=0, std=1):
        transform = None
        if flag == 0:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif flag == 1:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        return transform

    def torch_data_augment(self):
        transform_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])
        return transform_augment

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
