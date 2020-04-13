#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.base_name.task_name import TaskName


class PointCloudClassify(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.PC_Classify_Task)
        self.task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.model = self.torchModelProcess.modelTestInit(self.model)
        self.model.eval()

    def process(self, input_path):
        pass

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
        return output

    def postprocess(self, result):
        pass

    def compute_output(self, output_list):
        output = None
        loss_count = len(self.model.lossList)
        if loss_count == 1:
            output = self.model.lossList[0](output_list)
        return output
