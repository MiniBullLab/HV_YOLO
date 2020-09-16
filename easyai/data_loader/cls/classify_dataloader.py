#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.data_loader.cls.classify_dataset_process import ClassifyDatasetProcess
from easyai.data_loader.cls.classify_data_augment import ClassifyDataAugment


class ClassifyDataloader(TorchDataLoader):

    def __init__(self, train_path, resize_type, normalize_type,
                 mean=0, std=1, image_size=(416, 416),
                 data_channel=3, is_augment=False):
        super().__init__(data_channel)
        self.image_size = image_size
        self.is_augment = is_augment
        self.classify_sample = ClassifySample(train_path)
        self.classify_sample.read_sample(flag=0)
        self.dataset_process = ClassifyDatasetProcess(resize_type, normalize_type,
                                                      mean, std,
                                                      pad_color=self.get_pad_color())
        self.dataset_augment = ClassifyDataAugment(image_size)

    def __getitem__(self, index):
        img_path, label = self.classify_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        if self.is_augment:
            image = self.dataset_augment.augment(image)
            image = self.dataset_process.normalize_dataset(image)
        else:
            # image = self.dataset_process.normaliza_dataset(image, 0)
            image = self.dataset_process.normalize_dataset(image)
        return image, label

    def __len__(self):
        return self.classify_sample.get_sample_count()


def get_classify_train_dataloader(train_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.train_batch_size
    is_augment = data_config.train_data_augment
    dataloader = ClassifyDataloader(train_path, resize_type, normalize_type, mean, std,
                                    image_size, data_channel, is_augment=is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_classify_val_dataloader(val_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.train_batch_size
    dataloader = ClassifyDataloader(val_path, resize_type, normalize_type, mean, std,
                                    image_size, data_channel, is_augment=False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
