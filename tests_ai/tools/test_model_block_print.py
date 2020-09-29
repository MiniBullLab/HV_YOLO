#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_model_block_print.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.model_tool.model_block_print import model_print, backbone_model_print


def test_model_print(model):
    print("start...")
    model_print(model)
    print("End of game, have a nice day!")

def test_backbone_model_print(backbone):
    print("start...")
    backbone_model_print(backbone)
    print("End of game, have a nice day!")


if __name__ == '__main__':
    backbone = "../cfg/det2d/darknet53.cfg"
    model = "../cfg/cls/resnet_classify.cfg"

    test_model_print(model)
    # test_backbone_model_print(backbone)