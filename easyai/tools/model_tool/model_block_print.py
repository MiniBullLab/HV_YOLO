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
    input_x = torch.randn(1, 3, 224, 224)
    model_config = {"type": model_name}
    backbone = backbone_factory.get_backbone_model(model_config)
    if backbone is not None:
        backbone.print_block_name()
        # for k, value in backbone.named_parameters():
        #     print(k, value)


def model_print(model_name):
    model_factory = ModelFactory()
    input_x = torch.randn(1, 3, 32, 32)
    model_config = {"type": model_name}
    model = model_factory.get_model(model_config)
    if model is not None:
        model.print_block_name()
        # for k, value in model.named_children():
        #     print(k, value)


if __name__ == '__main__':
    options = ToolArgumentsParse.model_show_parse()
    if options.model is not None:
        model_print(options.model)
    elif options.backbone is not None:
        backbone_model_print(options.backbone)
    else:
        print("input param error")