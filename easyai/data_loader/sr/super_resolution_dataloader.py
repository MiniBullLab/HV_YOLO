#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.sr.super_resolution_sample import SuperResolutionSample
from easyai.data_loader.sr.super_resolution_dataset_process import SuperResolutionDatasetProcess


class SuperResolutionDataloader(data.Dataset):

    def __init__(self, train_path, image_size=(768, 320), data_channel=3, upscale_factor=3):
        super().__init__()
        self.image_size = image_size
        self.data_channel = data_channel
        self.upscale_factor = upscale_factor
        self.target_size = (image_size[0] * self.upscale_factor,
                            image_size[1] * self.upscale_factor)
        self.sr_sample = SuperResolutionSample(train_path)
        self.sr_sample.read_sample()
        self.image_process = ImageProcess()
        self.dataset_process = SuperResolutionDatasetProcess()

    def __getitem__(self, index):
        lr_path, hr_path = self.sr_sample.get_sample_path(index)
        src_lr_image = self.read_src_image(lr_path)
        src_hr_image = self.read_src_image(hr_path)
        image, target = self.dataset_process.resize_dataset(src_lr_image,
                                                            self.image_size,
                                                            src_hr_image,
                                                            self.target_size)
        image, target = self.dataset_process.normalize_dataset(image, target)
        torch_image = self.dataset_process.numpy_to_torch(image, flag=0)
        torch_target = self.dataset_process.numpy_to_torch(target, flag=0)
        return torch_image, torch_target

    def __len__(self):
        return self.sr_sample.get_sample_count()

    def read_src_image(self, image_path):
        src_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
        elif self.data_channel == 3:
            _, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("super resolution read src image error!")
        return src_image


def get_sr_train_dataloader(train_path, image_size, data_channel, upscale_factor,
                            batch_size, num_workers=8):
    dataloader = SuperResolutionDataloader(train_path, image_size,
                                           data_channel, upscale_factor)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_sr_val_dataloader(val_path, image_size, data_channel, upscale_factor,
                          batch_size, num_workers=8):
    dataloader = SuperResolutionDataloader(val_path, image_size,
                                           data_channel, upscale_factor)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
