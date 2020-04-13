#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.data_loader.seg.segment_data_augment import SegmentDataAugment


class SegmentDataLoader(data.Dataset):

    def __init__(self, train_path, image_size=(768, 320), is_augment=False):
        super().__init__()
        self.is_augment = is_augment
        self.image_size = image_size
        self.segment_sample = SegmentSample(train_path)
        self.segment_sample.read_sample()
        self.image_process = ImageProcess()
        self.dataset_process = SegmentDatasetProcess()
        self.data_augment = SegmentDataAugment()

        self.volid_label_seg = []
        self.valid_label_seg = [[0], [1]]

    def __getitem__(self, index):
        img_path, label_path = self.segment_sample.get_sample_path(index)
        src_image, rgb_image = self.image_process.readRgbImage(img_path)
        label = self.image_process.read_gray_image(label_path)
        rgb_image, target = self.dataset_process.resize_dataset(rgb_image,
                                                                self.image_size,
                                                                label,
                                                                self.volid_label_seg,
                                                                self.valid_label_seg)
        if self.is_augment:
            rgb_image, target = self.data_augment.augment(rgb_image, target)
        target = self.dataset_process.change_label(target, self.valid_label_seg)
        rgb_image = self.dataset_process.normaliza_dataset(rgb_image)
        torch_image = self.dataset_process.numpy_to_torch(rgb_image, flag=0)
        torch_target = self.dataset_process.numpy_to_torch(target).long()
        return torch_image, torch_target

    def __len__(self):
        return self.segment_sample.get_sample_count()


def get_segment_train_dataloader(train_path, image_size, batch_size, is_augment=False,
                                 num_workers=8):
    dataloader = SegmentDataLoader(train_path, image_size, is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_segment_val_dataloader(val_path, image_size, batch_size, num_workers=8):
    dataloader = SegmentDataLoader(val_path, image_size, False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
