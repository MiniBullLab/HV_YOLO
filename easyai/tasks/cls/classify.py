#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.visualization.classify_show import ClassifyShow
from easyai.base_name.task_name import TaskName


class Classify(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.Classify_Task)
        self.task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id,
                                                      default_args={
                                                          "data_channel": self.task_config.image_channel
                                                      })
        self.device = self.torchModelProcess.getDevice()
        self.result_show = ClassifyShow()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.model = self.torchModelProcess.modelTestInit(self.model)
        self.model.eval()

    def process(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size,
                                                self.task_config.image_channel)

        for index, (src_image, image) in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(image)
            result = self.postprocess(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))

            if not self.result_show.show(src_image,
                                         result,
                                         self.task_config.class_name):
                break

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

