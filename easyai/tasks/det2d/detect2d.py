#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.tasks.det2d.detect2d_result_process import Detect2dResultProcess
from easyai.base_algorithm.non_max_suppression import NonMaxSuppression
from easyai.base_algorithm.fast_non_max_suppression import FastNonMaxSuppression
from easyai.drawing.detect_show import DetectionShow
from easyai.base_name.task_name import TaskName


class Detection2d(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.Detect2d_Task)
        self.task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.torchModelProcess = TorchModelProcess()
        self.result_process = Detect2dResultProcess()
        self.nms_process = FastNonMaxSuppression()
        self.result_show = DetectionShow()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.src_size = (0, 0)

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.model = self.torchModelProcess.modelTestInit(self.model)
        self.model.eval()

    def process(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size)
        for i, (src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.set_src_size(src_image)

            self.timer.tic()
            result = self.infer(img, self.task_config.confidence_th)
            detection_objects = self.postprocess(result)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, detection_objects):
                break

    def save_result(self, filename, detection_objects):
        for object in detection_objects:
            confidence = object.classConfidence * object.objectConfidence
            x1 = object.min_corner.x
            y1 = object.min_corner.y
            x2 = object.max_corner.x
            y2 = object.max_corner.y
            temp_save_path = os.path.join(self.task_config.save_result_dir, "%s.txt" % object.name)
            with open(temp_save_path, 'a') as file:
                file.write("{} {} {} {} {} {}\n".format(filename, confidence, x1, y1, x2, y2))

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
            result = self.result_process.get_detection_result(output, threshold)
        return result

    def postprocess(self, result):
        detection_objects = self.nms_process.multi_class_nms(result, self.task_config.nms_th)
        detection_objects = self.result_process.resize_detection_objects(self.src_size,
                                                                         self.task_config.image_size,
                                                                         detection_objects,
                                                                         self.task_config.class_name)
        return detection_objects

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = prediction.squeeze(0)
        return prediction

    def set_src_size(self, src_data):
        shape = src_data.shape[:2]  # shape = [height, width]
        self.src_size = (shape[1], shape[0])



