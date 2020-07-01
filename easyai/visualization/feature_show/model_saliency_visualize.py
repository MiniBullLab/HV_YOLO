#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_model_process import TorchModelProcess


class ModelSaliencyVisualize():

    def __init__(self, is_backbone, input_size=(352, 640), data_channel=3,
                 class_number=100):
        self.is_backbone = is_backbone
        self.input_size = input_size
        self.class_number = class_number
        self.data_channel = data_channel
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()
        self.model_process = TorchModelProcess()

    def show_saliency_maps(self, image_path, model_name, weight_path):
        if self.is_backbone:
            model = self.backbone_factory.get_base_model(model_name,
                                                         default_args={"data_channel": self.data_channel})
            self.model_process.loadLatestModelWeight(weight_path, model)
        else:
            model = self.model_factory.get_model(model_name,
                                                 default_args={"data_channel": self.data_channel})
            self.model_process.loadLatestModelWeight(weight_path, model)

        image = load_image(image_path)
        backprop = Backprop(model)
        owl = apply_transforms(image, size=self.input_size)
        backprop.visualize(owl, self.class_number, guided=True)
