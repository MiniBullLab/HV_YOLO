#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import numpy as np
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class SegmentDatasetProcess(BaseDataSetProcess):
    def __init__(self):
        super().__init__()
        self.dataset_process = ImageDataSetProcess()
        self.image_pad_color = (0, 0, 0)
        self.label_pad_color = 250

    def normaliza_dataset(self, src_image):
        image = self.dataset_process.image_normaliza(src_image)
        image = self.dataset_process.numpy_transpose(image)
        return image

    def resize_dataset(self, src_image, image_size, label):
        image, _, _ = self.dataset_process.image_resize_square(src_image,
                                                               image_size,
                                                               color=self.image_pad_color)
        target, _, _ = self.dataset_process.image_resize_square(label,
                                                                image_size,
                                                                self.label_pad_color)
        target = np.array(label, dtype=np.uint8)
        return image, target

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
