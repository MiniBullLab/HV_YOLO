#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from easyai.model import ModelFactory
from onnx.model_show import ModelShow


def test_SR():
    model_factory = ModelFactory()
    show = ModelShow()
    input_x = torch.randn(1, 1, 72, 72)
    show.set_input(input_x)
    model = model_factory.get_model_from_name("MSRResNet")
    show.show_from_model(model)


def main():
    model_factory = ModelFactory()
    model = model_factory.get_model('../cfg/fgsegv2.cfg')
    for m in model.named_parameters():
        print(m)


if __name__ == '__main__':
    main()
    #test_SR()