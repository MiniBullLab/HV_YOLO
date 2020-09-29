#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import math
import random
import numpy as np
from easyai.helper.json_process import JsonProcess
from easyai.data_loader.utility.data_loader import DataLoader
from easyai.data_loader.multi_task.multi_task_sample import MultiTaskSample
from easyai.data_loader.multi_task.det2d_seg_data_augment import Det2dSegDataAugment
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.tools.sample.create_detection_sample import CreateDetectionSample


class Det2dSegTrainDataloader(DataLoader):

    def __init__(self, train_path, detect2d_class, seg_class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 batch_size=1, image_size=(768, 320), data_channel=3,
                 multi_scale=False, is_augment=False, balanced_sample=False):
        super().__init__(data_channel)
        self.detect2d_class = detect2d_class
        self.seg_number_class = len(seg_class_name)
        self.multi_scale = multi_scale
        self.is_augment = is_augment
        self.balanced_sample = balanced_sample
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_pad_color = (0, 0, 0)

        if balanced_sample:
            create_sample = CreateDetectionSample()
            save_sample_dir, _ = os.path.split(train_path)
            create_sample.createBalanceSample(train_path,
                                              save_sample_dir,
                                              detect2d_class)
        self.multi_task_sample = MultiTaskSample(train_path,
                                                 detect2d_class,
                                                 balanced_sample)
        self.multi_task_sample.read_sample()
        self.json_process = JsonProcess()
        self.image_dataset_process = ImageDataSetProcess()
        self.det2d_dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                             mean, std, self.get_pad_color())
        self.seg_dataset_process = SegmentDatasetProcess(resize_type, normalize_type,
                                                         mean, std, self.get_pad_color())
        self.dataset_augment = Det2dSegDataAugment()

        self.nF = self.multi_task_sample.get_sample_count()
        self.nB = math.ceil(self.nF / batch_size)  # number of batches

    def __iter__(self):
        self.count = -1
        self.multi_task_sample.shuffle_sample()
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        numpy_images = []
        numpy_boxes = []
        numpy_segments = []

        class_index = self.get_random_class()
        start_index = self.multi_task_sample.get_sample_start_index(self.count,
                                                                    self.batch_size,
                                                                    class_index)
        width, height = self.get_image_size()

        stop_index = start_index + self.batch_size
        for temp_index in range(start_index, stop_index):
            img_path, label_path, segment_path = self.multi_task_sample.get_sample_path(temp_index,
                                                                                        class_index)
            _, src_image = self.read_src_image(img_path)
            _, boxes = self.json_process.parse_rect_data(label_path)
            segment_label = self.image_process.read_gray_image(segment_path)

            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            ratio, pad_size = self.image_dataset_process.get_square_size(src_size, (width, height))
            image = self.image_dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                                   pad_color=self.image_pad_color)
            boxes = self.det2d_dataset_process.resize_labels(boxes, self.detect2d_class, ratio, pad_size)
            segment_label = self.seg_dataset_process.resize_lable(segment_label, ratio, pad_size)
            if self.is_augment:
                image, boxes, segments = self.dataset_augment.augment(image, boxes, segment_label)

            image = self.det2d_dataset_process.normalize_image(image)
            boxes = self.det2d_dataset_process.normalize_labels(boxes, (width, height))
            boxes = self.det2d_dataset_process.change_outside_labels(boxes)
            segment_label = self.seg_dataset_process.change_label(segment_label, self.seg_number_class)

            numpy_images.append(image)
            numpy_segments.append(segment_label)
            torch_boxes = self.det2d_dataset_process.numpy_to_torch(boxes, flag=0)
            numpy_boxes.append(torch_boxes)

        numpy_images = np.stack(numpy_images)
        numpy_segments = np.stack(numpy_segments)
        torch_images = self.all_numpy_to_tensor(numpy_images)
        torch_segments = self.seg_dataset_process.numpy_to_torch(numpy_segments).long()

        return torch_images, numpy_boxes, torch_segments

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


def get_det2d_seg_train_dataloader(train_path, data_config):
    detect2d_class = data_config.detect2d_class
    seg_class_name = data_config.segment_class
    image_size = data_config.image_size
    data_channel = data_config.image_channel
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    batch_size = data_config.train_batch_size
    dataloader = Det2dSegTrainDataloader(train_path, detect2d_class, seg_class_name,
                                         resize_type, normalize_type, mean, std,
                                         batch_size, image_size, data_channel,
                                         multi_scale=data_config.train_multi_scale,
                                         is_augment=data_config.train_data_augment,
                                         balanced_sample=data_config.balanced_sample)

    return dataloader
