#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: detect2d_test.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.seg.segment_train import SegmentionTrain


class SegTrainTask():

    def __init__(self, train_path, val_path, pretrain_model_path):
        self.train_path = train_path
        self.val_path = val_path
        self.pretrain_model_path = pretrain_model_path

    def segment_train(self, cfg_path, gpu_id, config_path):
        seg_train = SegmentionTrain(cfg_path, gpu_id, config_path)
        seg_train.load_pretrain_model(self.pretrain_model_path)
        seg_train.train(self.train_path, self.val_path)


def main(train_path, val_path, model, config_path, weights):
    print("process start...")
    train_task = SegTrainTask(train_path, val_path, weights)
    train_task.segment_train(model, 0, config_path)
    print("process end!")


if __name__ == '__main__':
    train_path = "/home/wfw/data/VOCdevkit/MagnetRing_segment/ImageSets/train.txt" # "/home/wfw/data/VOCdevkit/CULane/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/MagnetRing_segment/ImageSets/val.txt" # "/home/wfw/data/VOCdevkit/CULane/ImageSets/val.txt"
    model = "../cfg/seg/mobilenetv2_fgseg.cfg" # "../cfg/seg/mobilenetv2_fgseg.cfg" # mobilenet_fcn.cfg"
    config_path = "./tasks/seg/segmention_config.json"
    weights = "/home/wfw/EDGE/HV_YOLO/tests_ai/tasks/seg/segnet.pt" # "/home/wfw/HASCO/all_wights/mobilenetv2_FCN.pt"

    main(train_path, val_path, model, config_path, weights)
