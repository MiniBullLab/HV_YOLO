#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_create_detection_anchors.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.parameter.create_detection_anchors import CreateDetectionAnchors


def test(train_path, config_path, num_anchors):
    print("start...")
    test = CreateDetectionAnchors(train_path,
                                  config_path)
    test.get_anchors(num_anchors)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    train_path = "/home/wfw/data/VOCdevkit/BKLdata/ImageSets/train_val.txt"
    config_path = "../.log/config/detection2d_config_object.json"
    test(train_path, config_path, 9)