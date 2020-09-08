#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.det2d.detect2d_result_process import Detect2dResultProcess
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.base_algorithm.fast_non_max_suppression import FastNonMaxSuppression
from easyai.visualization.task_show.det2d_seg_drawing import Det2dSegTaskShow
from easyai.base_name.task_name import TaskName


class Det2dSegTask(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.Det2d_Seg_Task)
        self.det2d_result_process = Detect2dResultProcess()
        self.seg_result_process = SegmentResultProcess()
        self.nms_process = FastNonMaxSuppression()
        self.result_show = Det2dSegTaskShow()

        self.model = self.torchModelProcess.initModel(self.model_args, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.threshold_seg = 0.5  # binary class threshold

    def process(self, input_path, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size,
                                                self.task_config.image_channel)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.set_src_size(src_image)

            self.timer.tic()
            result_dets, result_seg = self.infer(img, self.task_config.confidence_th, self.threshold_seg)
            det2d_objects, segment_image = self.postprocess((result_dets, result_seg))
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, segment_image,
                                         self.task_config.segment_name, det2d_objects):
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

    def infer(self, input_data, threshold_det=0.0, threshold_seg=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output_dets, output_seg = self.compute_output(output_list)
            result_dets = self.det2d_result_process.get_detection_result(output_dets, threshold_det)
            result_seg = self.seg_result_process.get_segmentation_result(output_seg, threshold_seg)

        return result_dets, result_seg

    def postprocess(self, result):
        det2d_objects = self.nms_process.multi_class_nms(result[0], self.task_config.nms_th)
        det2d_objects = self.det2d_result_process.resize_detection_objects(self.src_size,
                                                                           self.task_config.image_size,
                                                                           det2d_objects,
                                                                           self.task_config.detect_name)
        segment_image = self.seg_result_process.resize_segmention_result(self.src_size,
                                                                         self.task_config.image_size,
                                                                         result)
        return det2d_objects, segment_image

    def compute_output(self, output_list):
        count = len(output_list)

        pred_seg = []
        temp = self.model.lossList[0](output_list[0])
        pred_seg.append(temp)
        prediction_seg = torch.cat(pred_seg, 1)
        prediction_seg = np.squeeze(prediction_seg.data.cpu().numpy())

        pred_dets = []
        for i in range(1, count):
            temp = self.model.lossList[i](output_list[i])
            pred_dets.append(temp)
        prediction_dets = torch.cat(pred_dets, 1)
        prediction_dets = prediction_dets.squeeze(0)
        return prediction_dets, prediction_seg
