#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
import math
import random
from easyai.helper import XMLProcess
from easyai.helper import ImageProcess
from easyai.data_loader.utility.data_loader import DataLoader
from easyai.data_loader.det.detection_sample import DetectionSample
from easyai.data_loader.det.detection_dataset_process import DetectionDataSetProcess
from easyai.data_loader.det.detection_data_augment import DetectionDataAugment


class DetectionTrainDataloader(DataLoader):

    def __init__(self, train_path, class_name, batch_size=1, image_size=(768, 320),
                 multi_scale=False, is_augment=False, balanced_sample=False):
        super().__init__()
        self.className = class_name
        self.multi_scale = multi_scale
        self.is_augment = is_augment
        self.balanced_sample = balanced_sample
        self.batch_size = batch_size
        self.image_size = image_size

        self.detection_sample = DetectionSample(train_path,
                                                class_name,
                                                balanced_sample)
        self.detection_sample.read_sample()
        self.xmlProcess = XMLProcess()
        self.image_process = ImageProcess()
        self.dataset_process = DetectionDataSetProcess()
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
            src_image, rgb_image = self.image_process.readRgbImage(img_path)
            _, _, boxes = self.xmlProcess.parseRectData(label_path)

            rgb_image, labels = self.dataset_process.resize_dataset(rgb_image,
                                                                    (width, height),
                                                                    boxes,
                                                                    self.className)
            rgb_image, labels = self.dataset_augment.augment(rgb_image, labels)
            rgb_image, labels = self.dataset_process.normaliza_dataset(rgb_image,
                                                                       labels,
                                                                       (width, height))

            labels = self.dataset_process.change_outside_labels(labels)

            numpy_images.append(rgb_image)

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
            class_index = np.random.randint(0, len(self.className))
            print("loading labels {}".format(self.className[class_index]))
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


