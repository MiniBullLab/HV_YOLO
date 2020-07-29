#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: detect2d_test.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.det2d.detect2d_train import Detection2dTrain


class TrainTask():

    def __init__(self, train_path, val_path, pretrain_model_path):
        self.train_path = train_path
        self.val_path = val_path
        self.pretrain_model_path = pretrain_model_path

    def detect2d_train(self, cfg_path, gpu_id, config_path):
        det2d_train = Detection2dTrain(cfg_path, gpu_id, config_path)
        det2d_train.load_pretrain_model(self.pretrain_model_path)
        det2d_train.train(self.train_path, self.val_path)


def main(trainPath, valPath, model, config_path, pretrainModel):
    print("process start...")
    train_task = TrainTask(trainPath, valPath, pretrainModel)
    train_task.detect2d_train(model, 0, config_path)
    print("process end!")


if __name__ == '__main__':
    train_path = "/home/wfw/data/VOCdevkit/BKLdata_json/ImageSets_xml/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/BKLdata_json/ImageSets_xml/val.txt"
    model = "YoloV3Det2d" # "/home/wfw/EDGE/HV_YOLO/tests_ai/tasks/det2d/yolov3_bkl.cfg" # YoloV3Det2d
    config_path = "./tasks/det2d/detection2d_YoloV3Det2d_config.json"
    weights = "/home/wfw/EDGE/HV_YOLO/tests_ai/tasks/det2d/yolov3_bkl.pt"

    main(train_path, val_path, model, config_path, weights)
