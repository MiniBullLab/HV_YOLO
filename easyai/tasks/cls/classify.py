#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.visualization.task_show.classify_show import ClassifyShow
from easyai.base_name.task_name import TaskName


class Classify(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Classify_Task)
        self.model_args['class_number'] = len(self.task_config.class_name)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id,
                                                      default_args=self.model_args)
        self.device = self.torchModelProcess.getDevice()
        self.result_show = ClassifyShow()

    def process(self, input_path):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size,
                                                self.task_config.image_channel)
        for index, (file_path, src_image, image) in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(image)
            result = self.postprocess(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
            self.save_result(file_path, result)
            if not self.result_show.show(src_image,
                                         result,
                                         self.task_config.class_name):
                break

    def save_result(self, file_path, class_index):
        path, filename_post = os.path.split(file_path[0])
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{} {}\n".format(filename_post, class_index))

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

