#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import abc
from easyai.helper.timer_process import TimerProcess
from easyai.data_loader.utility.images_loader import ImagesLoader
from easyai.data_loader.utility.video_loader import VideoLoader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.tasks.utility.base_task import BaseTask


class BaseInference(BaseTask):

    def __init__(self, config_path):
        super().__init__()
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.config_path = config_path
        self.model = None

    @abc.abstractmethod
    def process(self, input_path):
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

    def get_image_data_lodaer(self, input_path, image_size, data_channel):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path, image_size, data_channel)
        else:
            dataloader = VideoLoader(input_path, image_size, data_channel)
        return dataloader
