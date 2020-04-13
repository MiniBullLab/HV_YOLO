#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class ClassifyDatasetProcess(BaseDataSetProcess):

    def __init__(self, mean, std):
        super().__init__()
        self.dataset_process = ImageDataSetProcess()
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize_transform = self.torchvision_process.torch_normalize(flag=0,
                                                                            mean=self.mean,
                                                                            std=self.std)

    def normaliza_dataset(self, src_image, normaliza_type=0):
        result = None
        if normaliza_type == 0:  # numpy normalize
            normaliza_image = self.dataset_process.image_normaliza(src_image)
            image = self.dataset_process.numpy_normaliza(normaliza_image,
                                                         self.mean,
                                                         self.std)
            image = self.dataset_process.numpy_transpose(image, image.dtype)
            result = self.numpy_to_torch(image, flag=0)
        elif normaliza_type == 1:  # torchvision normalize
            result = self.normalize_transform(src_image)
        return result

    def resize_image(self, src_image, image_size):
        image = self.dataset_process.image_resize(src_image, image_size)
        return image
