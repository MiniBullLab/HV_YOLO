#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_compute_image_mean.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.parameter.compute_images_mean import ComputeImagesMean
from easyai.config.utility.config_factory import ConfigFactory


def test(task_name, config_path, input_path):
    print("start...")
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    test = ComputeImagesMean(image_size=task_config.image_size)
    mean, std = test.compute(input_path)
    print(mean, std)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    task_name = "classify"
    config_path = "../.log/config/classify_config.json"
    input_path = "/home/wfw/data/VOCdevkit/Tree_Ring_classify/ImageSets/train.txt"

    test(task_name, config_path, input_path)