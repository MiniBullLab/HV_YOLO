#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_model_summary.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.torch_utility.torch_summary import summary
from easyai.model.utility.model_factory import ModelFactory


def test(model, input_shape):
    print("start...")
    model_factory = ModelFactory()
    model = model_factory.get_model(model)
    summary(model, [1, 3, input_shape[1], input_shape[0]])
    print("End of game, have a nice day!")


if __name__ == '__main__':
    model = "../cfg/det2d/yolov3-coco.cfg"
    input_shape = [640, 352] # w, h

    test(model, input_shape)