#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import math
import random
import numpy as np
from easyai.helper.json_process import JsonProcess
from easyai.data_loader.utility.data_loader import DataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.det2d.det2d_data_augment import DetectionDataAugment
from easyai.tools.sample.create_detection_sample import CreateDetectionSample


class DetectionTrainDataloader(DataLoader):

    def __init__(self, train_path, detect2d_class,
                 resize_type, normalize_type, mean=0, std=1,
                 batch_size=1, image_size=(768, 320),
                 data_channel=3, multi_scale=False, is_augment=False, balanced_sample=False):
        super().__init__(data_channel)
        self.detect2d_class = detect2d_class
        self.multi_scale = multi_scale
        self.is_augment = is_augment
        self.balanced_sample = balanced_sample
        self.batch_size = batch_size
        self.image_size = image_size

        if balanced_sample:
            create_sample = CreateDetectionSample()
            save_sample_dir, _ = os.path.split(train_path)
            create_sample.createBalanceSample(train_path,
                                              save_sample_dir,
                                              detect2d_class)
        self.detection_sample = DetectionSample(train_path,
                                                detect2d_class,
                                                balanced_sample)
        self.detection_sample.read_sample()
        self.json_process = JsonProcess()

        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())
        self.dataset_augment = DetectionDataAugment()

        self.nF = self.detection_sample.get_sample_count()
        self.nB = math.ceil(self.nF / batch_size)  # number of batches

    def __iter__(self):
        self.count = -1
        self.detection_sample.shuffle_sample()
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        numpy_images = []
        numpy_labels = []

        class_index = self.get_random_class()
        start_index = self.detection_sample.get_sample_start_index(self.count,
                                                                   self.batch_size,
                                                                   class_index)
        width, height = self.get_image_size()

        stop_index = start_index + self.batch_size
        for temp_index in range(start_index, stop_index):
            img_path, label_path = self.detection_sample.get_sample_path(temp_index, class_index)
            _, src_image = self.read_src_image(img_path)
            _, boxes = self.json_process.parse_rect_data(label_path)

            image, labels = self.dataset_process.resize_dataset(src_image,
                                                                (width, height),
                                                                boxes,
                                                                self.detect2d_class)
            image, labels = self.dataset_augment.augment(image, labels)
            image = self.dataset_process.normalize_image(image)
            labels = self.dataset_process.normalize_labels(labels, (width, height))
            labels = self.dataset_process.change_outside_labels(labels)

            numpy_images.append(image)

            torch_labels = self.dataset_process.numpy_to_torch(labels, flag=0)
            numpy_labels.append(torch_labels)

        numpy_images = np.stack(numpy_images)
        torch_images = self.all_numpy_to_tensor(numpy_images)

        return torch_images, numpy_labels

    def __len__(self):
        return self.nB  # number of batches

    def get_random_class(self):
        class_index = None
        if self.balanced_sample:
            class_index = np.random.randint(0, len(self.detect2d_class))
            print("loading labels {}".format(self.detect2d_class[class_index]))
        return class_index

    def get_image_size(self):
        if self.multi_scale:
            # Multi-Scale YOLO Training
            print("wrong code for MultiScale")
            width = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            scale = float(self.image_size[0]) / float(self.image_size[1])
            height = int(round(float(width / scale) / 32.0) * 32)
        else:
            # Fixed-Scale YOLO Training
            width = self.image_size[0]
            height = self.image_size[1]
        return width, height


def get_detect2d_train_dataloader(train_path, data_config):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.train_batch_size
    dataloader = DetectionTrainDataloader(train_path, data_config.detect2d_class,
                                          resize_type, normalize_type, mean, std,
                                          batch_size, image_size, data_channel,
                                          multi_scale=data_config.train_multi_scale,
                                          is_augment=data_config.train_data_augment,
                                          balanced_sample=data_config.balanced_sample)
    return dataloader
