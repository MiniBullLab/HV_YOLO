#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import abc
from easyai.helper.timer_process import TimerProcess
from easyai.data_loader.utility.images_loader import ImagesLoader
from easyai.data_loader.utility.video_loader import VideoLoader
from easyai.tasks.utility.base_task import BaseTask


class BaseInference(BaseTask):

    def __init__(self, config_path):
        super().__init__()
        self.timer = TimerProcess()
        self.config_path = config_path

    @abc.abstractmethod
    def load_weights(self, weights_path):
        pass

    @abc.abstractmethod
    def process(self, input_path):
        pass

    @abc.abstractmethod
    def infer(self, input_data, threshold=0.0):
        pass

    @abc.abstractmethod
    def postprocess(self, result):
        pass

    def get_image_data_lodaer(self, input_path, image_size):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path, image_size)
        else:
            dataloader = VideoLoader(input_path, image_size)
        return dataloader
