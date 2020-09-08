#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_model_to_onnx.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.show_lr_scheduler import ShowLrScheduler


def test(task_name, config_path, epoch_iteration):
    print("start...")
    show_lr = ShowLrScheduler(task_name, config_path, epoch_iteration)
    show_lr.show()
    print("End of game, have a nice day!")


if __name__ == '__main__':
    task_name = "detect2d"
    config_path = "../.log/config/detection2d_config.json"
    epoch_iteration = 4000

    test(task_name, config_path, epoch_iteration)