#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.data_loader.cls.classify_dataset_process import ClassifyDatasetProcess
from easyai.data_loader.cls.classify_data_augment import ClassifyDataAugment


class ClassifyDataloader(data.Dataset):

    def __init__(self, train_path, mean=0, std=1, image_size=(416, 416),
                 data_channel=3, is_augment=False):
        self.image_size = image_size
        self.data_channel = data_channel
        self.is_augment = is_augment
        self.classify_sample = ClassifySample(train_path)
        self.classify_sample.read_sample(flag=0)
        self.image_process = ImageProcess()
        self.dataset_process = ClassifyDatasetProcess(mean, std)
        self.dataset_augment = ClassifyDataAugment()

    def __getitem__(self, index):
        img_path, label = self.classify_sample.get_sample_path(index)
        src_image = self.read_src_image(img_path)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        if self.is_augment:
            image = self.dataset_augment.augment(image)
            image = self.dataset_process.normalize_dataset(image, 1)
        else:
            # image = self.dataset_process.normaliza_dataset(image, 0)
            image = self.dataset_process.normalize_dataset(image, 1)
        return image, label

    def __len__(self):
        return self.classify_sample.get_sample_count()

    def read_src_image(self, image_path):
        src_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
        elif self.data_channel == 3:
            _, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("classify read src image error!")
        return src_image


def get_classify_train_dataloader(train_path, mean, std, image_size, data_channel,
                                  batch_size, is_augment=True, num_workers=8):
    dataloader = ClassifyDataloader(train_path, mean, std, image_size,
                                    data_channel, is_augment=is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_classify_val_dataloader(val_path, mean, std, image_size, data_channel,
                                batch_size, num_workers=8):
    dataloader = ClassifyDataloader(val_path, mean, std, image_size,
                                    data_channel, is_augment=False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
