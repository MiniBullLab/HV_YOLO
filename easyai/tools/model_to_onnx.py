#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.torch_to_onnx import TorchConvertOnnx
from easyai.config.utility.config_factory import ConfigFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class ModelConverter():

    def __init__(self, input_size=(352, 640), data_channel=3):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()
        self.converter = TorchConvertOnnx()
        self.input_size = input_size  # w * h
        self.data_channel = data_channel

    def model_convert(self, model_path, weight_path, save_dir):
        input_x = torch.randn(1, self.data_channel, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        self.converter.set_save_dir(save_dir)
        model = self.model_factory.get_model(model_path,
                                             default_args={"data_channel": self.data_channel})
        self.converter.torch2onnx(model, weight_path)

    def base_model_convert(self, base_model_path, weight_path, save_dir):
        input_x = torch.randn(1, self.data_channel, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        self.converter.set_save_dir(save_dir)
        model = self.backbone_factory.get_base_model(base_model_path,
                                                     default_args={"data_channel": self.data_channel})
        self.converter.torch2onnx(model, weight_path)


def main():
    pass


if __name__ == '__main__':
    options = ToolArgumentsParse.model_convert_parse()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(options.task_name, config_path=None)
    converter = ModelConverter(task_config.image_size)
    if options.model is not None:
        converter.model_convert(options.model, options.weight_path, options.save_dir)
    elif options.base_model is not None:
        converter.base_model_convert(options.backbone, options.weight_path, options.save_dir)
