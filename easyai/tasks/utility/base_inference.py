#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import abc
from pathlib import Path
from easyai.helper.timer_process import TimerProcess
from easyai.data_loader.utility.images_loader import ImagesLoader
from easyai.data_loader.utility.video_loader import VideoLoader
from easyai.data_loader.utility.text_data_loader import TextDataLoader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.tasks.utility.base_task import BaseTask


class BaseInference(BaseTask):

    def __init__(self, model_name, config_path, task_name):
        super().__init__()
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.config_path = config_path
        self.model = None
        self.src_size = (0, 0)
        self.task_config = self.config_factory.get_config(self.task_name, self.config_path)
        self.model_args = {"type": model_name,
                           "data_channel": self.task_config.data_channel
                           }

    @abc.abstractmethod
    def process(self, input_path, is_show=False):
        pass

    @abc.abstractmethod
    def infer(self, input_data, threshold=0.0):
        pass

    @abc.abstractmethod
    def postprocess(self, result):
        pass

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.model = self.torchModelProcess.modelTestInit(self.model)
        self.model.eval()

    def get_image_data_lodaer(self, input_path):
        if not os.path.exists(input_path):
            return None
        image_size = self.task_config.image_size
        data_channel = self.task_config.data_channel
        mean = self.task_config.data_mean
        std = self.task_config.data_std
        normalize_type = self.task_config.normalize_type
        resize_type = self.task_config.resize_type
        if Path(input_path).is_dir():
            dataloader = ImagesLoader(input_path, image_size, data_channel,
                                      resize_type, normalize_type, mean, std)
        elif Path(input_path).suffix in ['.txt', '.text']:
            dataloader = TextDataLoader(input_path, image_size, data_channel,
                                        resize_type, normalize_type, mean, std)
        else:
            dataloader = VideoLoader(input_path, image_size, data_channel,
                                     resize_type, normalize_type, mean, std)
        return dataloader

    def set_src_size(self, src_data):
        shape = src_data.shape[:2]  # shape = [height, width]
        self.src_size = (shape[1], shape[0])
