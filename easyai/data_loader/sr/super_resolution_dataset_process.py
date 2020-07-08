#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class SuperResolutionDatasetProcess(BaseDataSetProcess):

    def __init__(self):
        super().__init__()
        self.dataset_process = ImageDataSetProcess()

    def normalize_dataset(self, src_lr_image, src_hr_image):
        lr_image = self.dataset_process.image_normalize(src_lr_image)
        lr_image = self.dataset_process.numpy_transpose(lr_image)
        hr_image = self.dataset_process.image_normalize(src_hr_image)
        hr_image = self.dataset_process.numpy_transpose(hr_image)
        return lr_image, hr_image

    def resize_dataset(self, lr_image, src_image_size, hr_image, target_size):
        image = self.dataset_process.cv_image_resize(lr_image, src_image_size)
        target = self.dataset_process.cv_image_resize(hr_image, target_size)
        return image, target
