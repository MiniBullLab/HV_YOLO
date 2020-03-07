#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class ClassifyDatasetProcess():

    def __init__(self, mean, std):
        self.dataset_process = ImageDataSetProcess()
        self.mean = np.array(mean)
        self.std = np.array(std)

    def normaliza_dataset(self, src_image):
        normaliza_image = self.dataset_process.image_normaliza(src_image)
        image = self.dataset_process.numpy_normaliza(normaliza_image,
                                                     self.mean,
                                                     self.std)
        image = self.dataset_process.image_transpose(image)
        return image

    def resize_image(self, src_image, image_size):
        image = self.dataset_process.image_resize(src_image, image_size)
        return image
