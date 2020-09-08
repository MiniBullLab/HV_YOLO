#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: detect2d_test.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.cls.darknet import DarkNet
from easyai.torch_utility.torch_onnx.model_show import ModelShow


def darknet53(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 2, 8, 8, 4])
    model.set_name(BackboneName.Darknet53)
    return model

def main():
    print("start...")
    data_channel = 3
    backbone = darknet53(data_channel)
    show_process = ModelShow()
    show_process.show_from_model(backbone)
    print("End of game, have a nice day!")


if __name__ == '__main__':
    main()