#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_create_detection_sample.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.create_detection_sample import CreateDetectionSample


def test_balance_sample(config_path, train_path, output_dir):
    print("start...")
    test = CreateDetectionSample(config_path)
    test.createBalanceSample(train_path,
                             output_dir)
    print("End of game, have a nice day!")

def test_detection_sample(config_path, input_dir, output_path, probability):
    print("start...")
    test = CreateDetectionSample(config_path)
    test.createTrainAndTest(input_dir,
                            output_path,
                            probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    config_path = "../.log/config/detection2d_config.json"
    train_path = "/home/wfw/data/VOCdevkit/COCO/ImageSets/trainvalno5k.txt"
    output_dir = "/home/wfw/data/VOCdevkit/COCO/ImageSets"

    input_dir = "/home/wfw/data/VOCdevkit/COCO/JPEGImages"
    output_path = "/home/wfw/data/VOCdevkit/COCO/ImageSets"

    # test_balance_sample(config_path, train_path, output_dir)
    test_detection_sample(config_path, input_dir, output_path, 10)