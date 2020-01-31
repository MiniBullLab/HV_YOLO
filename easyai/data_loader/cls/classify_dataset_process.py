#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class ClassifyDatasetProcess():

    def __init__(self):
        self.dataset_process = ImageDataSetProcess()

    def normaliza_dataset(self, src_image):
        image = self.dataset_process.image_normaliza(src_image)
        image = self.dataset_process.image_transpose(image)
        return image

    def resize_image(self, src_image, image_size):
        image = self.dataset_process.image_resize(src_image, image_size)
        return image
