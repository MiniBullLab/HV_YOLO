#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import time
import torch
from evaluation.evaluatingOfmAp import *
from data_loader.det.detection_val_dataloader import get_detection_val_dataloader
from torch_utility.torch_model_process import TorchModelProcess
from base_algorithm.non_max_suppression import NonMaxSuppression
from tasks.det.detect_result_process import DetectResultProcess
from config import detect_config


class DetectionTest():

    def __init__(self, cfg_path, gpu_id):
        self.torchModelProcess = TorchModelProcess()
        self.result_process = DetectResultProcess()
        self.nms_process = NonMaxSuppression()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def test(self, val_Path):
        save_result_dir = os.path.join(detect_config.root_save_dir, 'results')
        os.system('rm -rf ' + save_result_dir)
        os.makedirs(save_result_dir, exist_ok=True)
        dataloader = get_detection_val_dataloader(val_Path, batch_size=1,
                                                  image_size=detect_config.imgSize)
        evaluator = MeanApEvaluating(val_Path, detect_config.className)

        prev_time = time.time()
        for i, (image_path, src_image, input_image) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            # Get detections
            with torch.no_grad():
                output_list = self.model(input_image.to(self.device))
                output = self.compute_output(output_list)
                result = self.result_process.get_detection_result(output, 5e-3)
            detection_objects = self.nms_process.multi_class_nms(result, detect_config.nmsThresh)
            detection_objects = self.result_process.resize_detection_objects(src_image,
                                                                             detect_config.imgSize,
                                                                             detection_objects,
                                                                             detect_config.className)
            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

            path, fileNameAndPost = os.path.split(image_path)
            fileName, post = os.path.splitext(fileNameAndPost)
            test_save_path = os.path.join(save_result_dir, 'comp4_det_test_')
            for object in detection_objects:
                confidence = object.classConfidence * object.objectConfidence
                x1 = object.min_corner.x
                y1 = object.min_corner.y
                x2 = object.max_corner.x
                y2 = object.max_corner.y
                temp_save_path = test_save_path + object.name + '.txt'
                with open(temp_save_path, 'a') as file:
                    file.write("{} {} {} {} {} {}\n".format(fileName, confidence, x1, y1, x2, y2))

        mAP, aps = evaluator.do_python_eval(save_result_dir, test_save_path)

        return mAP, aps

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = prediction.squeeze(0)
        return prediction

    def save_test_result(self, epoch, mAP, aps):
        # Write epoch results
        save_result_path = os.path.join(detect_config.root_save_dir, 'results.txt')
        with open(save_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(detect_config.className[i] + ": {:.3f} ".format(ap))
            file.write("\n")
