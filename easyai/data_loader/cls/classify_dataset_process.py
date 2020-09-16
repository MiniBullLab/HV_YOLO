#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class ClassifyDatasetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.use_torchvision = True
        self.torchvision_transform = self.torchvision_process.torch_normalize(flag=0,
                                                                              mean=self.mean,
                                                                              std=self.std)

    def normalize_dataset(self, src_image):
        if self.use_torchvision:
            result = self.torchvision_transform(src_image)
        else:
            image = self.dataset_process.normalize(input_data=src_image,
                                                   normalize_type=self.normalize_type,
                                                   mean=self.mean,
                                                   std=self.std)
            image = self.dataset_process.numpy_transpose(image, np.float32)
            result = self.numpy_to_torch(image, flag=0)
        return result

    def resize_image(self, src_image, image_size):
        image = self.dataset_process.resize(src_image, image_size,
                                            self.resize_type,
                                            pad_color=self.pad_color)
        return image
