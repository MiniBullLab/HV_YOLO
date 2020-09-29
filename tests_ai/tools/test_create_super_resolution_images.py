#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_create_segment_sample.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName
from easyai.tools.sample.create_super_resolution_images import CreateSuperResolutionImages


def test(config_path, input_path, upscale_factor):
    print("start...")
    test = CreateSuperResolutionImages()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.SuperResolution_Task, config_path=config_path)
    test.create_lr_images(input_path,
                          upscale_factor)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    config_path = ""
    input_path = "/home/wfw/data/VOCdevkit/DIV2k/HRImages"

    test(config_path, input_path, 2)