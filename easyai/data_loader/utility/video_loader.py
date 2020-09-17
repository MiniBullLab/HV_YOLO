#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper import VideoProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class VideoLoader(DataLoader):

    def __init__(self, video_path, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(data_channel)
        self.video_process = VideoProcess()
        self.dataset_process = ImageDataSetProcess()
        if not self.video_process.isVideoFile(video_path) or \
                not self.video_process.openVideo(video_path):
            raise Exception("Invalid path!", video_path)
        self.video_path = video_path
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.count = int(self.video_process.getFrameCount())
        self.image_pad_color = (0, 0, 0)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        success, cv_image = self.video_process.read_frame()

        if not success:
            raise StopIteration

        src_image = self.read_src_image(cv_image)
        image = self.dataset_process.resize(src_image, self.image_size, self.resize_type,
                                            pad_color=self.image_pad_color)
        image = self.dataset_process.normalize(input_data=image,
                                               normalize_type=self.normalize_type,
                                               mean=self.mean, std=self.std)
        numpy_image = self.dataset_process.numpy_transpose(image)
        torch_image = self.all_numpy_to_tensor(numpy_image)
        video_name = self.video_path + "_%d" % self.index
        return video_name, cv_image, torch_image

    def __len__(self):
        return self.count

    def read_src_image(self, cv_image):
        src_image = None
        if self.data_channel == 1:
            src_image = self.dataset_process.cv_image_color_convert(cv_image, 0)
        elif self.data_channel == 3:
            src_image = self.dataset_process.cv_image_color_convert(cv_image, 1)
        else:
            print("read src image error!")
        return cv_image, src_image
