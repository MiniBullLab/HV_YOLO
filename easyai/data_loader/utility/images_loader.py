#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper import ImageProcess
from easyai.helper import DirProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class ImagesLoader(DataLoader):

    def __init__(self, input_dir, image_size=(416, 416)):
        super().__init__()
        self.image_size = image_size
        self.imageProcess = ImageProcess()
        self.dirProcess = DirProcess()
        self.dataset_process = ImageDataSetProcess()
        temp_files = self.dirProcess.getDirFiles(input_dir, "*.*")
        self.files = list(temp_files)
        self.count = len(self.files)
        self.color = (127.5, 127.5, 127.5)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.count:
            raise StopIteration
        image_path = self.files[self.index]

        # Read image
        srcImage, rgb_image = self.imageProcess.readRgbImage(image_path)

        # Padded resize
        rgb_image, _, _ = self.dataset_process.image_resize_square(rgb_image,
                                                                   self.image_size,
                                                                   self.color)
        rgb_image = self.dataset_process.image_normaliza(rgb_image)
        numpy_image = self.dataset_process.numpy_transpose(rgb_image)
        torch_image = self.all_numpy_to_tensor(numpy_image)
        return srcImage, torch_image

    def __len__(self):
        return self.count
