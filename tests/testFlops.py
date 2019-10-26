#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from torch_utility.model_summary import summary
from model.model_factory import ModelFactory


def main():
    print("process start...")
    model_factory = ModelFactory()
    model = model_factory.get_model("../cfg/yolov3.cfg")
    summary(model, [1, 3, 640, 352])
    print("process end!")

if __name__ == '__main__':
    main()