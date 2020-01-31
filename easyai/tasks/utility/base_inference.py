#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.helper.timer_process import TimerProcess
from easyai.data_loader.utility.images_loader import ImagesLoader
from easyai.data_loader.utility.video_loader import VideoLoader


class BaseInference():

    def __init__(self):
        self.timer = TimerProcess()

    def get_image_data_lodaer(self, input_path, image_size):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path, image_size)
        else:
            dataloader = VideoLoader(input_path, image_size)
        return dataloader
