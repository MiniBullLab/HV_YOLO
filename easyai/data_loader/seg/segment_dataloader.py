#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.data_loader.seg.segment_data_augment import SegmentDataAugment
from easyai.tools.sample.convert_segment_label import ConvertSegmentionLable


class SegmentDataLoader(TorchDataLoader):

    def __init__(self, train_path, class_names, label_type,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(768, 320), data_channel=3, is_augment=False):
        super().__init__(data_channel)
        self.class_names = class_names
        self.label_type = label_type
        self.number_class = len(class_names)
        self.is_augment = is_augment
        self.image_size = image_size
        self.segment_sample = SegmentSample(train_path)
        self.segment_sample.read_sample()
        self.dataset_process = SegmentDatasetProcess(resize_type, normalize_type,
                                                     mean, std, self.get_pad_color())
        self.data_augment = SegmentDataAugment()
        self.label_converter = ConvertSegmentionLable()

    def __getitem__(self, index):
        img_path, label_path = self.segment_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        label = self.read_label_image(label_path)
        image, target = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            label)
        if self.is_augment:
            image, target = self.data_augment.augment(image, target)
        target = self.dataset_process.change_label(target, self.number_class)
        rgb_image = self.dataset_process.normalize_dataset(image)
        torch_image = self.dataset_process.numpy_to_torch(rgb_image, flag=0)
        torch_target = self.dataset_process.numpy_to_torch(target).long()
        return torch_image, torch_target

    def __len__(self):
        return self.segment_sample.get_sample_count()

    def read_label_image(self, label_path):
        if self.label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.label_converter.process_segment_label(label_path,
                                                              self.label_type,
                                                              self.class_names)
        return mask


def get_segment_train_dataloader(train_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    class_names = data_config.segment_class
    label_type = data_config.seg_label_type
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.train_batch_size
    is_augment = data_config.train_data_augment
    dataloader = SegmentDataLoader(train_path, class_names, label_type,
                                   resize_type, normalize_type, mean, std,
                                   image_size, data_channel, is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_segment_val_dataloader(val_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    class_names = data_config.segment_class
    label_type = data_config.seg_label_type
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.test_batch_size
    dataloader = SegmentDataLoader(val_path, class_names, label_type,
                                   resize_type, normalize_type, mean, std,
                                   image_size, data_channel, False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
