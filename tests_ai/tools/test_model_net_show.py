#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_model_net_show.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.model_tool.model_net_show import ModelNetShow


def test_model_show(model):
    print("start...")
    show = ModelNetShow()
    show.model_show(model)
    print("End of game, have a nice day!")

def test_backbone_show(backbone):
    print("start...")
    show = ModelNetShow()
    show.backbone_show(backbone)
    print("End of game, have a nice day!")

def test_onnx_show(onnx_path):
    print("start...")
    show = ModelNetShow()
    show.onnx_show(onnx_path)
    print("End of game, have a nice day!")


if __name__ == '__main__':
    model = "../cfg/det2d/yolov3-coco.cfg"
    backbone = "../cfg/det2d/darknet53.cfg"
    onnx_path = "./yolov3-coco.onnx"

    # test_model_show(model)
    test_backbone_show(backbone)
    # test_onnx_show(onnx_path)