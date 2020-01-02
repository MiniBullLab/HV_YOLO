#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import os
import sys
import time
import cv2
import torch
from easyAI.evaluation.evaluatingOfmAp import *
from easyAI.data_loader.imageDetectValDataLoader import ImageDetectValDataLoader
from easyAI.data_loader.trainDataProcess import TrainDataProcess
from easyAI.model.modelResultProcess import ModelResultProcess
from easyAI.torch_utility.torchModelProcess import TorchModelProcess
from easyAI.base_algorithm.nms import non_max_suppression
from easyAI.config import detectConfig
from easyAI.helper.arguments_parse import ArgumentsParse


class DetectionTest():

    def __init__(self, cfg_path, gpu_id):
        self.trainDataProcess = TrainDataProcess()
        self.torchModelProcess = TorchModelProcess()
        self.modelResultProcess = ModelResultProcess()
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def test(self, val_Path):
        save_result_dir = os.path.join(detectConfig.root_save_dir, 'results')
        os.system('rm -rf ' + save_result_dir)
        os.makedirs(save_result_dir, exist_ok=True)
        dataloader = ImageDetectValDataLoader(val_Path, batch_size=1, img_size=detectConfig.imgSize)
        evaluator = MeanApEvaluating(val_Path, detectConfig.className)

        prev_time = time.time()
        for i, (img_path, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            # Get detections
            with torch.no_grad():
                output = self.model(img.to(self.device))
                pred = self.modelResultProcess.detectResult(self.model, output, 5e-3)
                # print(pred)
                detections = non_max_suppression(pred, 5e-3, detectConfig.nmsThresh)

            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

            path, fileNameAndPost = os.path.split(img_path)
            fileName, post = os.path.splitext(fileNameAndPost)
            img = cv2.imread(img_path)

            detectObjects = self.trainDataProcess.resizeDetectObjects(img, detectConfig.imgSize, detections,
                                                                 detectConfig.className)
            test_save_path = os.path.join(save_result_dir, 'comp4_det_test_')
            for object in detectObjects:
                confidence = object.classConfidence * object.objectConfidence
                x1 = object.min_corner.x
                y1 = object.min_corner.y
                x2 = object.max_corner.x
                y2 = object.max_corner.y
                temp_save_path = test_save_path + object.name + '.txt'
                with open(temp_save_path, 'a') as file:
                    file.write(
                        "{} {} {} {} {} {}\n".format(fileName, confidence, x1, y1, x2, y2))

        mAP, aps = evaluator.do_python_eval(save_result_dir, test_save_path)

        return mAP, aps

    def save_test_result(self, epoch, mAP, aps):
        # Write epoch results
        save_result_path = os.path.join(detectConfig.root_save_dir, 'results.txt')
        with open(save_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(detectConfig.className[i] + ": {:.3f} ".format(ap))
            file.write("\n")


def main():
    print("process start...")
    options = ArgumentsParse.test_input_parse()
    detect_test = DetectionTest(options.cfg, 0)
    detect_test.load_weights(options.weights)
    detect_test.test(options.valPath)
    print("process end!")


if __name__ == '__main__':
    main()
