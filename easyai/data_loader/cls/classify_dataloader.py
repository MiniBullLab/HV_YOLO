#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.data_loader.cls.classify_dataset_process import ClassifyDatasetProcess
from easyai.data_loader.cls.classify_data_augment import ClassifyDataAugment


class ClassifyDataloader(data.Dataset):

    def __init__(self, train_path, mean=0, std=1,
                 image_size=(416, 416), is_augment=False):
        self.image_size = image_size
        self.is_augment = is_augment
        self.classify_sample = ClassifySample(train_path)
        self.classify_sample.read_sample(flag=0)
        self.image_process = ImageProcess()
        self.dataset_process = ClassifyDatasetProcess(mean, std)
        self.dataset_augment = ClassifyDataAugment()

    def __getitem__(self, index):
        img_path, label = self.classify_sample.get_sample_path(index)
        src_image, rgb_image = self.image_process.readRgbImage(img_path)
        rgb_image = self.dataset_process.resize_image(rgb_image, self.image_size)
        if self.is_augment:
            rgb_image = self.dataset_augment.augment(rgb_image)
            rgb_image = self.dataset_process.normaliza_dataset(rgb_image, 1)
        else:
            # rgb_image = self.dataset_process.normaliza_dataset(rgb_image, 0)
            rgb_image = self.dataset_process.normaliza_dataset(rgb_image, 1)
        return rgb_image, label

    def __len__(self):
        return self.classify_sample.get_sample_count()


def get_classify_train_dataloader(train_path, mean, std, image_size,
                                  batch_size, is_augment=True, num_workers=8):
    dataloader = ClassifyDataloader(train_path, mean, std, image_size, is_augment=is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_classify_val_dataloader(val_path, mean, std, image_size,
                                batch_size, num_workers=8):
    dataloader = ClassifyDataloader(val_path, mean, std, image_size, is_augment=False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
