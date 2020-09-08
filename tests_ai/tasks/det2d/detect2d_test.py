#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: detect2d_test.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.det2d.detect2d_test import Detection2dTest


class Detect2dTestTask():

    def __init__(self, val_path, weight_path):
        self.val_path = val_path
        self.weight_path = weight_path

    def detect2d_task(self,  cfg_path, gpu_id, config_path):
        det2d_test = Detection2dTest(cfg_path, gpu_id, config_path)
        det2d_test.load_weights(self.weight_path)
        det2d_test.test(self.val_path)

def main(valPath, model, config_path, weights):
    print("process start...")
    test_task = Detect2dTestTask(valPath, weights)
    test_task.detect2d_task(model, 0, config_path)
    print("process end!")


if __name__ == '__main__':
    val_path = "/home/wfw/data/VOCdevkit/COCO/ImageSets/val.txt"
    model = "../cfg/det2d/yolov3-coco.cfg"
    config_path = "./tasks/det2d/detection2d_config_COCO.json"
    weights = "/home/wfw/HASCO/all_wights/yolov3_coco.pt"

    main(val_path, model, config_path, weights)
