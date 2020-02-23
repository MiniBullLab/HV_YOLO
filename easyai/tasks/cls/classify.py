#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.config import classify_config


class Classify(BaseInference):

    def __init__(self, cfg_path, gpu_id):
        super().__init__()
        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def process(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                classify_config.imgSize)

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
        return output

    def postprocess(self, result):
        class_indices = torch.argmax(result, dim=1)
        return class_indices

    def compute_output(self, output_list):
        output = None
        if len(output_list) == 1:
            output = self.model.lossList[0](output_list[0])
        return output

