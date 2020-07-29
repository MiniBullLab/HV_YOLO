#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: detect2d_test.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.seg.segment import Segmentation


class SegInferenceTask():

    def __init__(self, input_path, weight_path):
        self.input_path = input_path
        self.weight_path = weight_path

    def segment_task(self, cfg_path, gpu_id, config_path):
        seg = Segmentation(cfg_path, gpu_id, config_path)
        seg.load_weights(self.weight_path)
        seg.process(self.input_path)

def main(input_path, model, config_path, weights):
    print("process start...")
    inference_task = SegInferenceTask(input_path, weights)
    inference_task.segment_task(model, 0, config_path)
    print("process end!")


if __name__ == '__main__':
    input_path = "/home/wfw/data/VOCdevkit/MagnetRing_segment/JPEGImages"
    model = "../cfg/seg/mobilenetv2_fgseg.cfg"
    config_path = "./tasks/seg/segmention_config.json"
    weights = ".log/snapshot/seg_best.pt"

    main(input_path, model, config_path, weights)
