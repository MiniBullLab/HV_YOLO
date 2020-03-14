#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from optparse import OptionParser
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory


def parse_arguments():

    parser = OptionParser()
    parser.description = "This program print model block name"

    parser.add_option("-b", "--base_model", dest="base_model",
                      action="store", type="string", default=None,
                      help="base model name or cfg file path")

    (options, args) = parser.parse_args()
    return options


def main(base_model):
    backbone_factory = BackboneFactory()
    input_x = torch.randn(1, 3, 32, 32)
    model = backbone_factory.get_base_model(base_model)
    model.print_block_name()


if __name__ == '__main__':
    options = parse_arguments()
    main(options.base_model)