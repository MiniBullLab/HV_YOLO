#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.multi_task.multi_task_sample import MultiTaskSample
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess


class Det2dSegValDataloader(TorchDataLoader):

    def __init__(self, val_path, detect2d_class, seg_class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3):
        super().__init__(data_channel)
        self.detect2d_class = detect2d_class
        self.seg_number_class = len(seg_class_name)
        self.image_size = image_size
        self.image_pad_color = (0, 0, 0)

        self.multi_task_sample = MultiTaskSample(val_path,
                                                 detect2d_class,
                                                 False)
        self.multi_task_sample.read_sample()

        self.image_dataset_process = ImageDataSetProcess()
        self.det2d_dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                             mean, std, self.get_pad_color())
        self.seg_dataset_process = SegmentDatasetProcess(resize_type, normalize_type,
                                                         mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, label_path, segment_path = self.multi_task_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
        segment_label = self.image_process.read_gray_image(segment_path)
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        ratio, pad_size = self.image_dataset_process.get_square_size(src_size, self.image_size)
        image = self.image_dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                               pad_color=self.image_pad_color)
        segment_label = self.seg_dataset_process.resize_lable(segment_label, ratio, pad_size)
        segment_label = self.seg_dataset_process.change_label(segment_label, self.seg_number_class)

        image = self.det2d_dataset_process.normalize_image(image)
        image = self.det2d_dataset_process.numpy_to_torch(image, flag=0)

        torch_segment = self.seg_dataset_process.numpy_to_torch(segment_label).long()

        return img_path, cv_image, image, torch_segment

    def __len__(self):
        return self.multi_task_sample.get_sample_count()


def get_det2d_seg_val_dataloader(val_path, data_config, num_workers=8):
    detect2d_class = data_config.detect2d_class
    seg_class_name = data_config.segment_class
    image_size = data_config.image_size
    data_channel = data_config.image_channel
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    batch_size = 1
    dataloader = Det2dSegValDataloader(val_path, detect2d_class, seg_class_name,
                                       resize_type, normalize_type, mean, std,
                                       image_size, data_channel)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
