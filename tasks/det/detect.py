#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from tasks.utility.base_inference import BaseInference
from torch_utility.torch_model_process import TorchModelProcess
from tasks.det.detect_result_process import DetectResultProcess
from base_algorithm.non_max_suppression import NonMaxSuppression
from drawing.detect_show import DetectionShow
from config import detect_config


class Detection(BaseInference):

    def __init__(self, cfg_path, gpu_id):
        super().__init__()
        self.torchModelProcess = TorchModelProcess()
        self.result_process = DetectResultProcess()
        self.nms_process = NonMaxSuppression()
        self.result_show = DetectionShow()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def detect(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                detect_config.imgSize)
        for i, (src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.timer.tic()
            with torch.no_grad():
                output_list = self.model(img.to(self.device))
                output = self.compute_output(output_list)
                result = self.result_process.get_detection_result(output, detect_config.confThresh)
            detection_objects = self.nms_process.multi_class_nms(result, detect_config.nmsThresh)
            detection_objects = self.result_process.resize_detection_objects(src_image,
                                                                             detect_config.imgSize,
                                                                             detection_objects,
                                                                             detect_config.className)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, detection_objects):
                break

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = prediction.squeeze(0)
        return prediction



