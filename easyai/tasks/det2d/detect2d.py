#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.det2d.detect2d_result_process import Detect2dResultProcess
from easyai.base_algorithm.fast_non_max_suppression import FastNonMaxSuppression
from easyai.visualization.task_show.detect2d_show import DetectionShow
from easyai.base_name.task_name import TaskName


class Detection2d(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Detect2d_Task)

        self.result_process = Detect2dResultProcess()
        self.nms_process = FastNonMaxSuppression()
        self.result_show = DetectionShow()

        self.model_args['type'] = cfg_path
        self.model_args['class_number'] = len(self.task_config.class_name)
        self.model = self.torchModelProcess.initModel(self.model_args, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def process(self, input_path):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path,
                                                self.task_config.image_size,
                                                self.task_config.image_channel)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.set_src_size(src_image)

            self.timer.tic()
            result = self.infer(img, self.task_config.confidence_th)
            detection_objects = self.postprocess(result)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))
            self.save_result(file_path, detection_objects, 0)
            if not self.result_show.show(src_image, detection_objects):
                break

    def save_result(self, file_path, detection_objects, flag=0):
        path, filename_post = os.path.split(file_path)
        if flag == 0:
            save_data = filename_post + "|"
            for temp_object in detection_objects:
                confidence = temp_object.classConfidence * temp_object.objectConfidence
                x1 = temp_object.min_corner.x
                y1 = temp_object.min_corner.y
                x2 = temp_object.max_corner.x
                y2 = temp_object.max_corner.y
                save_data = save_data + "{} {} {} {} {}|".format(temp_object.name,
                                                                 confidence,
                                                                 x1, y1, x2, y2)
            save_data += "\n"
            with open(self.task_config.save_result_path, 'a') as file:
                file.write(self.task_config.save_result_path)
        elif flag == 1:
            for temp_object in detection_objects:
                confidence = temp_object.classConfidence * temp_object.objectConfidence
                x1 = temp_object.min_corner.x
                y1 = temp_object.min_corner.y
                x2 = temp_object.max_corner.x
                y2 = temp_object.max_corner.y
                temp_save_path = os.path.join(self.task_config.save_result_dir, "%s.txt" % temp_object.name)
                with open(temp_save_path, 'a') as file:
                    file.write("{} {} {} {} {} {}\n".format(filename_post, confidence, x1, y1, x2, y2))

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
            result = self.result_process.get_detection_result(output, threshold,
                                                              self.task_config.post_prcoess_type)
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



