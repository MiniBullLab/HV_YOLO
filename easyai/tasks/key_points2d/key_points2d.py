#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.key_points2d.key_points2d_result_process import KeyPoints2dResultProcess
from easyai.visualization.task_show.key_points2d_show import KeyPointsShow
from easyai.base_name.task_name import TaskName


class KeyPoints2d(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.KeyPoints2d_Task)

        self.result_process = KeyPoints2dResultProcess(self.task_config.points_count)
        self.result_show = KeyPointsShow()

        self.model_args['type'] = cfg_path
        self.model = self.torchModelProcess.initModel(self.model_args, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def process(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size,
                                                self.task_config.image_channel)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.set_src_size(src_image)

            self.timer.tic()
            result = self.infer(img, self.task_config.confidence_th)
            result_objects = self.postprocess(result)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, result_objects):
                break

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
            result = self.result_process.get_keypoints_result(output, threshold,
                                                              self.task_config.post_prcoess_type)
        return result

    def postprocess(self, result):
        result_objects = self.result_process.resize_keypoints_objects(self.src_size,
                                                                      self.task_config.image_size,
                                                                      result,
                                                                      self.task_config.class_name)
        return result_objects

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = prediction.squeeze(0)
        return prediction
