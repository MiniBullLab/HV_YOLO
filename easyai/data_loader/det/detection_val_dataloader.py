#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.det.detection_sample import DetectionSample
from easyai.data_loader.det.detection_dataset_process import DetectionDataSetProcess


class DetectionValDataLoader(data.Dataset):

    def __init__(self, val_path, class_name, image_size=(416, 416), data_channel=3):
        super().__init__()
        self.image_size = image_size
        self.data_channel = data_channel
        self.detection_sample = DetectionSample(val_path,
                                                class_name,
                                                False)
        self.detection_sample.read_sample()
        self.image_process = ImageProcess()
        self.dataset_process = DetectionDataSetProcess()

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
        image, _ = self.dataset_process.resize_dataset(src_image,
                                                       self.image_size)
        image, _ = self.dataset_process.normaliza_dataset(image)
        image = self.dataset_process.numpy_to_torch(image, flag=0)
        return img_path, cv_image, image

    def __len__(self):
        return self.detection_sample.get_sample_count()

    def read_src_image(self, image_path):
        src_image = None
        cv_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
            cv_image = src_image[:]
        elif self.data_channel == 3:
            cv_image, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("det2d read src image error!")
        return cv_image, src_image


def get_detection_val_dataloader(val_path, class_name, image_size, data_channel,
                                 batch_size, num_workers=8):
    dataloader = DetectionValDataLoader(val_path, class_name, image_size, data_channel)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
