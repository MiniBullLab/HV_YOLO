#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess


class DetectionValDataLoader(TorchDataLoader):

    def __init__(self, val_path, class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3):
        super().__init__(data_channel)
        self.image_size = image_size
        self.detection_sample = DetectionSample(val_path,
                                                class_name,
                                                False)
        self.detection_sample.read_sample()
        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
        image = self.dataset_process.resize_src_image(src_image,
                                                      self.image_size)
        image = self.dataset_process.normalize_image(image)
        image = self.dataset_process.numpy_to_torch(image, flag=0)
        return img_path, cv_image, image

    def __len__(self):
        return self.detection_sample.get_sample_count()


def get_detection_val_dataloader(val_path, data_config, num_workers=8):
    class_name = data_config.detect2d_class
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = 1
    dataloader = DetectionValDataLoader(val_path, class_name,
                                        resize_type, normalize_type, mean, std,
                                        image_size, data_channel)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
