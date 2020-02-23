#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.drawing.segment_show import SegmentionShow
from easyai.config import segment_config


class Segmentation(BaseInference):

    def __init__(self, cfg_path, gpu_id):
        super().__init__()
        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.result_process = SegmentResultProcess()
        self.result_show = SegmentionShow()

        self.threshold = 0.5
        self.src_size = (0, 0)

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def process(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                segment_config.imgSize)
        for index, (src_image, image) in enumerate(dataloader):
            self.timer.tic()
            self.set_src_size(src_image)
            prediction = self.infer(image, self.threshold)
            result = self.postprocess(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
            if not self.result_show.show(src_image, result,
                                         segment_config.className):
                break

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
            prediction = self.result_process.get_detection_result(output, threshold)
        return prediction

    def postprocess(self, result):
        result = self.result_process.resize_segmention_result(self.src_size,
                                                              segment_config.imgSize,
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

    def set_src_size(self, src_data):
        shape = src_data.shape[:2]  # shape = [height, width]
        self.src_size = (shape[1], shape[0])
