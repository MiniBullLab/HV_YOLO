#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.helper.json_process import JsonProcess
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.key_point2d.key_point2d_dataset_process import KeyPoint2dDataSetProcess
from easyai.data_loader.utility.batch_dataset_merge import detection_data_merge


class KeyPoint2dDataLoader(data.Dataset):

    def __init__(self, data_path, class_name,
                 image_size=(416, 416), data_channel=3,
                 points_count=9):
        super().__init__()
        self.class_name = class_name
        self.image_size = image_size
        self.data_channel = data_channel
        self.detection_sample = DetectionSample(data_path,
                                                class_name,
                                                False)
        self.detection_sample.read_sample()

        self.json_process = JsonProcess()
        self.image_process = ImageProcess()
        self.dataset_process = KeyPoint2dDataSetProcess(points_count)

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
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


def get_key_points2d_train_dataloader(train_path, class_name,
                                      image_size, data_channel, points_count,
                                      batch_size, num_workers=8):
    dataloader = KeyPoint2dDataLoader(train_path, class_name,
                                      image_size, data_channel,
                                      points_count)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True,
                             collate_fn=detection_data_merge)
    return result


def get_key_points2d_val_dataloader(val_path, class_name,
                                    image_size, data_channel, points_count,
                                    batch_size, num_workers=8):
    dataloader = KeyPoint2dDataLoader(val_path, class_name,
                                      image_size, data_channel,
                                      points_count)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False,
                             collate_fn=detection_data_merge)
    return result



