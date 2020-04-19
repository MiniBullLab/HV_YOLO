#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper import VideoProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class VideoLoader(DataLoader):

    def __init__(self, video_path, image_size=(416, 416)):
        super().__init__()
        self.video_process = VideoProcess()
        self.dataset_process = ImageDataSetProcess()
        if not self.video_process.isVideoFile(video_path) or \
                not self.video_process.openVideo(video_path):
            raise Exception("Invalid path!", video_path)
        self.image_size = image_size
        self.count = int(self.video_process.getFrameCount())
        self.color = (127.5, 127.5, 127.5)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        success, src_image, rgb_image = self.video_process.readRGBFrame()

        if not success:
            raise StopIteration

        # padded resize
        rgb_image, _, _ = self.dataset_process.image_resize_square(rgb_image,
                                                                   self.image_size,
                                                                   self.color)
        rgb_image = self.dataset_process.image_normaliza(rgb_image)
        numpy_image = self.dataset_process.numpy_transpose(rgb_image)
        torch_image = self.all_numpy_to_tensor(numpy_image, 0)
        return src_image, torch_image

    def __len__(self):
        return self.count
