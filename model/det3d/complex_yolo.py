#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_block.utility_layer import RouteLayer
from base_block.utility_block import ConvBNActivationBlock
from base_block.darknet_block import ReorgBlock
from base_name.block_name import BatchNormType, ActivationType, BlockType
from base_model.utility.base_model_factory import BaseModelFactory
from base_model.utility.base_model import *
from base_name.model_name import ModelName


class ComplexYOLO(BaseModel):

    def __init__(self):
        super().__init__()
        self.set_name(ModelName.ComplexYOLO)
        self.bn_name = BatchNormType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.factory = BaseModelFactory()
        self.lossList = []
        self.out_channels = []
        self.index = 0
        self.create_model()

    def create_model(self):
        basic_model = self.factory.get_base_model("./cfg/complex-darknet19.cfg")
        base_out_channels = basic_model.getOutChannelList()
        self.add_block_list(BlockType.BaseNet, basic_model, base_out_channels[-1])

        input_channel = self.out_channels[-1]
        output_channel = 1024
        conv1 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = 1024
        conv2 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, output_channel)

        layer1 = RouteLayer('12')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        reorg = ReorgBlock()
        output_channel = reorg.stride * reorg.stride * self.out_channels[-1]
        self.add_block_list(reorg.get_name(), reorg, output_channel)

        layer2 = RouteLayer('-3,-1')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = 1024
        conv3 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = 75
        conv4 = nn.Conv2d(in_channels=input_channel,
                          out_channels=output_channel,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.add_block_list(BlockType.Convolutional, conv4, output_channel)

    def create_loss(self):
        pass

    def add_block_list(self, block_name, block, output_channel):
        block_name = "%s_%d" % (block_name, self.index)
        self.add_module(block_name, block)
        self.index += 1
        self.out_channels.append(output_channel)

    def freeze_bn(self, is_freeze):
        for m in self.modules():
            if is_freeze and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.ShortcutLayer in key:
                x = block(layer_outputs)
            else:
                x = block(x)
            layer_outputs.append(x)
        return layer_outputs
