#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from pathlib import Path
from easyai.helper import ImageProcess
from easyai.helper import DirProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class TextDataLoader(DataLoader):

    def __init__(self, input_path, image_size=(416, 416), data_channel=3):
        super().__init__()
        if Path(input_path).suffix in ['.txt', '.text']:
            raise Exception("Invalid path!", input_path)
        self.image_size = image_size
        self.data_channel = data_channel
        self.image_process = ImageProcess()
        self.dirProcess = DirProcess()
        self.dataset_process = ImageDataSetProcess()
        self.files = self.get_image_list(input_path)
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
        shape = src_image.shape[:2]  # shape = [height, width]
        src_size = (shape[1], shape[0])
        ratio, pad_size = self.dataset_process.get_square_size(src_size, self.image_size)
        image = self.dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                         color=self.image_pad_color)
        image = self.dataset_process.image_normalize(image)
        numpy_image = self.dataset_process.numpy_transpose(image)
        torch_image = self.all_numpy_to_tensor(numpy_image)
        return image_path, cv_image, torch_image

    def __len__(self):
        return self.count

    def get_image_list(self, input_path):
        result = []
        path, _ = os.path.split(input_path)
        images_dir = os.path.join(path, "../JPEGImages")
        for line_data in self.dirProcess.getFileData(input_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) > 0:
                image_path = os.path.join(images_dir, data_list[0])
                # print(image_path)
                if os.path.exists(image_path):
                    result.append(image_path)
                else:
                    print("%s not exist" % image_path)
            else:
                print("% error" % line_data)
        return result

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
