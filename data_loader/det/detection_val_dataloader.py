#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import torch.utils.data as data
from helper.imageProcess import ImageProcess
from data_loader.det.detection_sample import DetectionSample
from data_loader.det.detection_dataset_process import DetectionDataSetProcess


class DetectionValDataLoader(data.Dataset):

    def __init__(self, val_path, image_size=(416, 416)):
        super().__init__()
        self.image_size = image_size
        self.detection_sample = DetectionSample(False, val_path, None)
        self.detection_sample.read_sample()
        self.image_process = ImageProcess()
        self.dataset_process = DetectionDataSetProcess()

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        src_image, rgb_image = self.image_process.readRgbImage(img_path)
        rgb_image, _ = self.dataset_process.resize_dataset(rgb_image,
                                                           self.image_size)
        rgb_image, _ = self.dataset_process.normaliza_dataset(rgb_image)
        rgb_image = torch.from_numpy(rgb_image)
        return img_path, src_image, rgb_image

    def __len__(self):
        return self.detection_sample.get_sample_count()


def get_detection_val_dataloader(val_path, image_size, batch_size, num_workers=8):
    dataloader = DetectionValDataLoader(val_path, image_size)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result