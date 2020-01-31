#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BatchNormType, ActivationType
from easyai.base_name.base_model_name import BaseModelName
from easyai.model.base_block.utility_block import ConvBNActivationBlock, InvertedResidual


__all__ = ['mobilenetv2_1_0', 'MobileNetV2']


class MobileNetV2(BaseModel):

    def __init__(self, data_channel=3, num_blocks=[1, 2, 3, 4, 3, 3, 1],
                 out_channels=[16, 24, 32, 64, 96, 160, 320], strides=[1, 2, 2, 2, 1, 2, 1],
                 dilations=[1, 1, 1, 1, 1, 1, 1], bnName=BatchNormType.BatchNormalize2d,
                 activationName=ActivationType.ReLU, expand_ratios=[1, 6, 6, 6, 6, 6, 6]):
        super().__init__()
        self.set_name(BaseModelName.MobileNetV2_1_0)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.expand_ratios = expand_ratios
        self.first_output = 32

        self.outChannelList = []
        self.index = 0

        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_mobile_layer(self.out_channels[index], self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.bnName, self.activationName,
                                   self.expand_ratios[index])
            self.in_channels = self.outChannelList[-1]

    def make_mobile_layer(self, out_channels, num_blocks, stride, dilation,
                          bnName, activationName, expand_ratio):
        if dilation > 1:
            stride = 1
        down_layers = InvertedResidual(self.in_channels, out_channels, stride=stride,
                                       expand_ratio=expand_ratio, dilation=dilation,
                                       bnName=bnName, activationName=activationName)
        name = "down_%s" % down_layers.get_name()
        self.addBlockList(name, down_layers, out_channels)
        for _ in range(num_blocks - 1):
            layer = InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=expand_ratio,
                                     dilation=dilation, bnName=bnName, activationName=activationName)
            self.addBlockList(layer.get_name(), layer, out_channels)

    def addBlockList(self, blockName, block, out_channel):
        blockName = "base_%s_%d" % (blockName, self.index)
        self.add_module(blockName, block)
        self.outChannelList.append(out_channel)
        self.index += 1

    def getOutChannelList(self):
        return self.outChannelList

    def printBlockName(self):
        for key in self._modules.keys():
            print(key)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def mobilenetv2_1_0():
    model = MobileNetV2(num_blocks=[1, 2, 3, 4, 3, 3, 1])
    model.set_name(BaseModelName.MobileNetV2_1_0)
    return model


def mobilenetv2_1_0_dilated8():
    model = MobileNetV2(num_blocks=[1, 2, 3, 4, 3, 3, 1],
                        dilations=[1, 1, 1, 2, 2, 4, 4])
    model.set_name(BaseModelName.MobileNetV2_1_0)
    return model


