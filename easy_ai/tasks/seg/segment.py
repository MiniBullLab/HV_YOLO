#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import numpy as np
from easy_ai.tasks.utility.base_inference import BaseInference
from easy_ai.torch_utility.torch_model_process import TorchModelProcess
from easy_ai.tasks.seg.segment_result_process import SegmentResultProcess
from easy_ai.drawing.segment_show import SegmentionShow
from easy_ai.config import segment_config


class Segmentation(BaseInference):

    def __init__(self, cfg_path, gpu_id):
        super().__init__()
        self.torchModelProcess = TorchModelProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.result_process = SegmentResultProcess()
        self.result_show = SegmentionShow()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def segment(self, input_path):
        dataloader = self.get_image_data_lodaer(input_path,
                                                segment_config.imgSize)
        for index, (src_image, image) in enumerate(dataloader):
            self.timer.tic()
            with torch.no_grad():
                output_list = self.model(image.to(self.device))
                output = self.compute_output(output_list)
                prediction = self.result_process.get_detection_result(output)

            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))

            result = self.result_process.resize_segmention_result(src_image,
                                                                  segment_config.imgSize,
                                                                  prediction)
            if not self.result_show.show(src_image, result,
                                         segment_config.className):
                break

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = np.squeeze(prediction.data.cpu().numpy())
        return prediction
