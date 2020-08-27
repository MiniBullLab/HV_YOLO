#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.visualization.task_show.segment_show import SegmentionShow
from easyai.helper.imageProcess import ImageProcess
from easyai.base_name.task_name import TaskName


class Segmentation(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Segment_Task)

        self.model_args['class_number'] = len(self.task_config.class_name)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id,
                                                      default_args=self.model_args)
        self.device = self.torchModelProcess.getDevice()

        self.result_process = SegmentResultProcess()

        self.result_show = SegmentionShow()

        self.image_process = ImageProcess()

        self.threshold = 0.5  # binary class threshold

    def process(self, input_path):
        os.system('rm -rf ' + self.task_config.save_result_dir)
        os.makedirs(self.task_config.save_result_dir, exist_ok=True)

        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size,
                                                self.task_config.image_channel)
        for index, (file_path, src_image, image) in enumerate(dataloader):
            self.timer.tic()
            self.set_src_size(src_image)
            prediction, _ = self.infer(image, self.threshold)
            result = self.postprocess(prediction)
            self.save_result(file_path, result)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
            if not self.result_show.show(src_image, result,
                                         self.task_config.class_name):
                break

    def save_result(self, file_path, prediction):
        path, filename_post = os.path.split(file_path)
        filename, post = os.path.splitext(filename_post)
        save_result_path = os.path.join(self.task_config.save_result_dir, "%s.png" % filename)
        self.image_process.opencv_save_image(save_result_path, prediction)

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list[:])
            prediction = self.result_process.get_segmentation_result(output, threshold)
        return prediction, output_list

    def postprocess(self, result):
        result = self.result_process.resize_segmention_result(self.src_size,
                                                              self.task_config.image_size,
                                                              result)
        return result

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = np.squeeze(prediction.data.cpu().numpy())
        return prediction

