#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper import DirProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class ImagesLoader(DataLoader):

    def __init__(self, input_dir, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(data_channel)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = mean
        self.std = std
        self.resize_type = resize_type
        self.image_process = ImageProcess()
        self.dirProcess = DirProcess()
        self.dataset_process = ImageDataSetProcess()
        temp_files = self.dirProcess.getDirFiles(input_dir, "*.*")
        self.files = list(temp_files)
        self.count = len(self.files)
        self.image_pad_color = (0, 0, 0)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.count:
            raise StopIteration
        image_path = self.files[self.index]
        cv_image, src_image = self.read_src_image(image_path)
        image = self.dataset_process.resize(src_image, self.image_size, self.resize_type,
                                            pad_color=self.image_pad_color)
        image = self.dataset_process.normalize(input_data=image,
                                               normalize_type=self.normalize_type,
                                               mean=self.mean, std=self.std)
        numpy_image = self.dataset_process.numpy_transpose(image)
        torch_image = self.all_numpy_to_tensor(numpy_image)
        return image_path, cv_image, torch_image

    def __len__(self):
        return self.count
