#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.shufflenetv2 import ShuffleNetV2
from base_model.mobilenetv2 import MobileNetV2
from base_model.resnet import ResNet
from base_model.base_model_factory import BaseModelFactory

def main():
    print("process start...")
    base_model_factory = BaseModelFactory()
    #model = base_model_factory.get_base_model_from_cfg("../cfg/darknet-tiny.cfg")
    model = base_model_factory.get_base_model("../cfg/vgg16.cfg")
    model.printBlockName()
    print("process end!")

if __name__ == '__main__':
    main()