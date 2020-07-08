#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.multi_task.multi_task_sample import MultiTaskSample
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess


class Det2dSegValDataloader(data.Dataset):

    def __init__(self, val_path, det2d_class_name, seg_class_name,
                 image_size=(416, 416), data_channel=3):
        super().__init__()
        self.det2d_class_name = det2d_class_name
        self.seg_number_class = len(seg_class_name)
        self.image_size = image_size
        self.data_channel = data_channel
        self.image_pad_color = (0, 0, 0)

        self.multi_task_sample = MultiTaskSample(val_path,
                                                 det2d_class_name,
                                                 False)
        self.multi_task_sample.read_sample()

        self.image_process = ImageProcess()
        self.image_dataset_process = ImageDataSetProcess()
        self.det2d_dataset_process = DetectionDataSetProcess(self.image_pad_color)
        self.seg_dataset_process = SegmentDatasetProcess(self.image_pad_color)

    def __getitem__(self, index):
        img_path, label_path, segment_path = self.multi_task_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
        segment_label = self.image_process.read_gray_image(segment_path)
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        ratio, pad_size = self.image_dataset_process.get_square_size(src_size, self.image_size)
        image = self.image_dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                               color=self.image_pad_color)
        segment_label = self.seg_dataset_process.resize_lable(segment_label, ratio, pad_size)
        segment_label = self.seg_dataset_process.change_label(segment_label, self.seg_number_class)

        image = self.det2d_dataset_process.normalize_image(image)
        image = self.det2d_dataset_process.numpy_to_torch(image, flag=0)

        torch_segment = self.seg_dataset_process.numpy_to_torch(segment_label).long()

        return img_path, cv_image, image, torch_segment

    def __len__(self):
        return self.multi_task_sample.get_sample_count()

    def read_src_image(self, image_path):
        src_image = None
        cv_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
            cv_image = src_image[:]
        elif self.data_channel == 3:
            cv_image, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("det2d_seg read src image error!")
        return cv_image, src_image


def get_det2d_seg_val_dataloader(val_path, det2d_class_name, seg_class_name,
                                 image_size, data_channel, batch_size, num_workers=8):
    dataloader = Det2dSegValDataloader(val_path, det2d_class_name, seg_class_name,
                                       image_size, data_channel)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
