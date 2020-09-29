#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_model_to_onnx.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.model_tool.model_to_onnx import ModelConverter
from easyai.config.utility.config_factory import ConfigFactory


def test_model_convert(task_name, config_path, model, weight_path, save_dir):
    print("start...")
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    converter = ModelConverter(task_config.image_size)
    converter.model_convert(model, weight_path, save_dir)
    print("End of game, have a nice day!")

def test_base_model_convert(task_name, config_path, backbone, weight_path, save_dir):
    print("start...")
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    converter = ModelConverter(task_config.image_size)
    converter.base_model_convert(backbone, weight_path, save_dir)
    print("End of game, have a nice day!")


if __name__ == '__main__':
    task_name = "detect2d"
    config_path = "../.log/config/detection2d_config.json"
    model = "../cfg/det2d/yolov3-coco.cfg"
    weight_path = None
    save_dir = "./"

    test_model_convert(task_name, config_path, model, weight_path, save_dir)