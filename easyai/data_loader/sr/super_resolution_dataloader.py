#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.sr.super_resolution_sample import SuperResolutionSample
from easyai.data_loader.sr.super_resolution_dataset_process import SuperResolutionDatasetProcess


class SuperResolutionDataloader(data.Dataset):

    def __init__(self, train_path, image_size=(768, 320), upscale_factor=3):
        super().__init__()
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.target_size = (image_size[0] * self.upscale_factor,
                            image_size[1] * self.upscale_factor)
        self.sr_sample = SuperResolutionSample(train_path)
        self.sr_sample.read_sample()
        self.image_process = ImageProcess()
        self.dataset_process = SuperResolutionDatasetProcess()

    def __getitem__(self, index):
        img_path, label_path = self.sr_sample.get_sample_path(index)
        gray_image = self.image_process.read_gray_image(img_path)
        label = self.image_process.read_gray_image(label_path)
        gray_image, target = self.dataset_process.resize_dataset(gray_image,
                                                                 self.image_size,
                                                                 label,
                                                                 self.target_size)
        gray_image = self.dataset_process.normaliza_dataset(gray_image)
        torch_image = self.dataset_process.numpy_to_torch(gray_image, flag=0)
        torch_target = self.dataset_process.numpy_to_torch(target).long()
        return torch_image, torch_target

    def __len__(self):
        return self.sr_sample.get_sample_count()


def get_sr_train_dataloader(train_path, image_size, upscale_factor,
                            batch_size, num_workers=8):
    dataloader = SuperResolutionDataloader(train_path, image_size, upscale_factor)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_sr_val_dataloader(val_path, image_size, upscale_factor,
                          batch_size, num_workers=8):
    dataloader = SuperResolutionDataloader(val_path, image_size, upscale_factor)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
