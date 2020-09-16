#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from torch.utils.data import Dataset
from easyai.helper.imageProcess import ImageProcess


class TorchDataLoader(Dataset):

    def __init__(self, data_channel):
        super().__init__()
        self.data_channel = data_channel
        self.image_process = ImageProcess()

    def read_src_image(self, image_path):
        src_image = None
        cv_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
            cv_image = src_image[:]
        elif self.data_channel == 3:
            cv_image, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("dataloader src image error!")
        return cv_image, src_image

    def get_pad_color(self):
        result = None
        if self.data_channel == 1:
            result = 0
        elif self.data_channel == 3:
            result = (0, 0, 0)
        return result

