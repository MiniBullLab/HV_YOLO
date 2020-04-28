#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.data_loader.seg.segment_data_augment import SegmentDataAugment


class SegmentDataLoader(data.Dataset):

    def __init__(self, train_path, number_class, image_size=(768, 320),
                 data_channel=3, is_augment=False):
        super().__init__()
        self.number_class = number_class
        self.is_augment = is_augment
        self.image_size = image_size
        self.data_channel = data_channel
        self.segment_sample = SegmentSample(train_path)
        self.segment_sample.read_sample()
        self.image_process = ImageProcess()
        self.dataset_process = SegmentDatasetProcess()
        self.data_augment = SegmentDataAugment()

    def __getitem__(self, index):
        img_path, label_path = self.segment_sample.get_sample_path(index)
        src_image = self.read_src_image(img_path)
        label = self.image_process.read_gray_image(label_path)
        image, target = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            label)
        if self.is_augment:
            image, target = self.data_augment.augment(image, target)
        target = self.dataset_process.change_label(target, self.number_class)
        rgb_image = self.dataset_process.normaliza_dataset(image)
        torch_image = self.dataset_process.numpy_to_torch(rgb_image, flag=0)
        torch_target = self.dataset_process.numpy_to_torch(target).long()
        return torch_image, torch_target

    def __len__(self):
        return self.segment_sample.get_sample_count()

    def read_src_image(self, image_path):
        src_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
        elif self.data_channel == 3:
            _, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("segment read src image error!")
        return src_image


def get_segment_train_dataloader(train_path, number_class, image_size, data_channel,
                                 batch_size, is_augment=False, num_workers=8):
    dataloader = SegmentDataLoader(train_path, number_class, image_size, data_channel, is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_segment_val_dataloader(val_path, number_class, image_size, data_channel,
                               batch_size, num_workers=8):
    dataloader = SegmentDataLoader(val_path, number_class, image_size, data_channel, False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
