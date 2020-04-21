#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class SuperResolutionDatasetProcess(BaseDataSetProcess):

    def __init__(self):
        super().__init__()
        self.dataset_process = ImageDataSetProcess()

    def normaliza_dataset(self, src_image):
        image = self.dataset_process.numpy_transpose(src_image)
        return image

    def resize_dataset(self, src_image, src_image_size, label, target_size):
        image = self.dataset_process.image_resize(src_image, src_image_size)
        target = self.dataset_process.image_resize(label, target_size)
        return image, target
