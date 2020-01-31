#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easy_ai.helper.imageProcess import ImageProcess
from easy_ai.data_loader.cls.classify_sample import ClassifySample
from easy_ai.data_loader.cls.classify_dataset_process import ClassifyDatasetProcess


class ClassifyDataloader(data.Dataset):

    def __init__(self, train_path, image_size=(416, 416)):
        self.image_size = image_size
        self.classify_sample = ClassifySample(train_path)
        self.classify_sample.read_sample(flag=0)
        self.image_process = ImageProcess()
        self.dataset_process = ClassifyDatasetProcess()

    def __getitem__(self, index):
        img_path, label = self.classify_sample.get_sample_path(index)
        src_image, rgb_image = self.image_process.readRgbImage(img_path)
        rgb_image = self.dataset_process.resize_image(rgb_image, self.image_size)
        rgb_image = self.dataset_process.normaliza_dataset(rgb_image)
        return rgb_image, label

    def __len__(self):
        return self.classify_sample.get_sample_count()


def get_classify_train_dataloader(train_path, image_size, batch_size, num_workers=8):
    dataloader = ClassifyDataloader(train_path, image_size)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_classify_val_dataloader(val_path, image_size, batch_size, num_workers=8):
    dataloader = ClassifyDataloader(val_path, image_size)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result
