#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper import ImageProcess
from easyai.helper import DirProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class ImagesLoader(DataLoader):

    def __init__(self, input_dir, image_size=(416, 416), data_channel=3):
        super().__init__()
        self.image_size = image_size
        self.data_channel = data_channel
        self.image_process = ImageProcess()
        self.dirProcess = DirProcess()
        self.dataset_process = ImageDataSetProcess()
        temp_files = self.dirProcess.getDirFiles(input_dir, "*.*")
        self.files = list(temp_files)
        self.count = len(self.files)
        self.color = (0, 0, 0)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.count:
            raise StopIteration
        image_path = self.files[self.index]
        cv_image, src_image = self.read_src_image(image_path)
        # Padded resize
        image, _, _ = self.dataset_process.image_resize_square(src_image,
                                                               self.image_size,
                                                               self.color)
        image = self.dataset_process.image_normaliza(image)
        numpy_image = self.dataset_process.numpy_transpose(image)
        torch_image = self.all_numpy_to_tensor(numpy_image)
        return cv_image, torch_image

    def __len__(self):
        return self.count

    def read_src_image(self, image_path):
        src_image = None
        cv_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
            cv_image = src_image[:]
        elif self.data_channel == 3:
            cv_image, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("read src image error!")
        return cv_image, src_image
