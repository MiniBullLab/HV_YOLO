#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


def backbone_model_print(model_name):
    backbone_factory = BackboneFactory()
    input_x = torch.randn(1, 3, 32, 32)
    backbone = backbone_factory.get_base_model(model_name)
    backbone.print_block_name()


def model_print(model_name):
    model_factory = ModelFactory()
    input_x = torch.randn(1, 3, 32, 32)
    model = model_factory.get_model(model_name)
    model.print_block_name()


if __name__ == '__main__':
    options = ToolArgumentsParse.model_show_parse()
    if options.model is not None:
        model_print(options.model)
    elif options.base_model is not None:
        backbone_model_print(options.base_model)
    else:
        print("input param error")