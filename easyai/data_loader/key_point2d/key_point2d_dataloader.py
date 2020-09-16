#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.json_process import JsonProcess
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.key_point2d.key_point2d_dataset_process import KeyPoint2dDataSetProcess
from easyai.data_loader.utility.batch_dataset_merge import detection_data_merge


class KeyPoint2dDataLoader(TorchDataLoader):

    def __init__(self, data_path, class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3,
                 points_count=9):
        super().__init__(data_channel)
        self.class_name = class_name
        self.image_size = image_size
        self.detection_sample = DetectionSample(data_path,
                                                class_name,
                                                False)
        self.detection_sample.read_sample()

        self.json_process = JsonProcess()
        self.dataset_process = KeyPoint2dDataSetProcess(points_count, resize_type, normalize_type,
                                                        mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        _, boxes = self.json_process.parse_key_points_data(label_path)
        image, labels = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            boxes,
                                                            self.class_name)
        image = self.dataset_process.normalize_image(image)
        labels = self.dataset_process.normalize_labels(labels, self.image_size)
        labels = self.dataset_process.change_outside_labels(labels)

        torch_label = self.dataset_process.numpy_to_torch(labels, flag=0)
        torch_image = self.dataset_process.numpy_to_torch(image, flag=0)
        return torch_image, torch_label

    def __len__(self):
        return self.detection_sample.get_sample_count()


def get_key_points2d_train_dataloader(train_path, data_config, num_workers=8):
    class_name = data_config.points_class
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    points_count = data_config.points_count
    batch_size = data_config.train_batch_size
    dataloader = KeyPoint2dDataLoader(train_path, class_name,
                                      resize_type, normalize_type, mean, std,
                                      image_size, data_channel,
                                      points_count)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True,
                             collate_fn=detection_data_merge)
    return result


def get_key_points2d_val_dataloader(val_path, data_config, num_workers=8):
    class_name = data_config.points_class
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    points_count = data_config.points_count
    batch_size = 1
    dataloader = KeyPoint2dDataLoader(val_path, class_name,
                                      resize_type, normalize_type, mean, std,
                                      image_size, data_channel,
                                      points_count)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False,
                             collate_fn=detection_data_merge)
    return result



