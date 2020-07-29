#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_convert_segment_label.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.convert_segment_label import ConvertSegmentionLable
from easyai.config.utility.config_factory import ConfigFactory


def test(task_name, config_path, input_path):
    print("start...")
    test = ConvertSegmentionLable()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    test.convert_segment_label(input_path,
                               task_config.label_is_gray,
                               task_config.class_name)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    task_name = "segment"
    config_path = "../.log/config/segmention_config.json"
    input_path = "/home/wfw/data/VOCdevkit/LedScratch_segment/SegmentLabel_a"

    test(task_name, config_path, input_path)