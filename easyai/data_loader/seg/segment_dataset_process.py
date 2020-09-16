#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import numpy as np
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class SegmentDatasetProcess(TaskDataSetProcess):
    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.label_pad_color = 250

    def normalize_dataset(self, src_image):
        image = self.dataset_process.normalize(input_data=src_image,
                                               normalize_type=self.normalize_type,
                                               mean=self.mean,
                                               std=self.std)
        image = self.dataset_process.numpy_transpose(image)
        return image

    def resize_dataset(self, src_image, image_size, label):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        ratio, pad_size = self.dataset_process.get_square_size(src_size, image_size)
        image = self.dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                         pad_color=self.pad_color)
        target = self.resize_lable(label, ratio, pad_size)
        return image, target

    def resize_lable(self, label, ratio, pad_size):
        target = self.dataset_process.image_resize_square(label, ratio, pad_size,
                                                          self.label_pad_color)
        target = np.array(target, dtype=np.uint8)
        return target

    def change_label(self, label, number_class):
        valid_masks = np.zeros(label.shape)
        for index in range(0, number_class):
            valid_mask = label == index  # set false to position of seg that not in valid label
            valid_masks += valid_mask  # set 0 to position of seg that not in valid label
        valid_masks[valid_masks == 0] = -1
        mask = np.float32(label) * valid_masks
        mask[mask < 0] = self.label_pad_color
        mask = np.uint8(mask)
        return mask
