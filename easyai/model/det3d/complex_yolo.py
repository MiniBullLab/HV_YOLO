#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.darknet_block import ReorgBlock
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.base_model import *


class ComplexYOLO(BaseModel):

    def __init__(self):
        super().__init__()
        self.set_name(ModelName.ComplexYOLO)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        basic_model = self.factory.get_base_model("./cfg/det3d/complex-darknet19.cfg")
        base_out_channels = basic_model.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, basic_model, base_out_channels[-1])

        input_channel = self.block_out_channels[-1]
        output_channel = 1024
        conv1 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, output_channel)

        input_channel = self.block_out_channels[-1]
        output_channel = 1024
        conv2 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, output_channel)

        layer1 = RouteLayer('12')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        reorg = ReorgBlock()
        output_channel = reorg.stride * reorg.stride * self.out_channels[-1]
        self.add_block_list(reorg.get_name(), reorg, output_channel)

        layer2 = RouteLayer('-3,-1')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.block_out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        input_channel = self.block_out_channels[-1]
        output_channel = 1024
        conv3 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, output_channel)

        input_channel = self.block_out_channels[-1]
        output_channel = 75
        conv4 = nn.Conv2d(in_channels=input_channel,
                          out_channels=output_channel,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.add_block_list(LayerType.Convolutional, conv4, output_channel)

    def create_loss(self, input_dict=None):
        self.lossList = []

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            else:
                x = block(x)
            layer_outputs.append(x)
        return layer_outputs
