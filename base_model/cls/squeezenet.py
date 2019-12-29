#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_model.utility.base_model import *
from base_name.base_model_name import BaseModelName
from base_name.block_name import BatchNormType, ActivationType, BlockType
from base_block.utility_block import ConvActivationBlock
from base_block.squeezenet_block import FireBlock


__all__ = ['SqueezeNet', 'DilatedSqueezeNet']


class SqueezeNet(BaseModel):

    def __init__(self, data_channel=3, bnName=BatchNormType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super(SqueezeNet, self).__init__()
        self.set_name(BaseModelName.SqueezeNet)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 64

        self.outChannelList = []
        self.index = 0

        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvActivationBlock(in_channels=self.data_channel,
                                     out_channels=self.first_output,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     dilation=1,
                                     activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.addBlockList(BlockType.MyMaxPool2d, layer2, self.first_output)

        planes = (16, 64, 64)
        fire1 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire1.get_name(), fire1, output_channle)

        planes = (16, 64, 64)
        fire2 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire2.get_name(), fire2, output_channle)

        layer3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.addBlockList(BlockType.MyMaxPool2d, layer3, output_channle)

        planes = (32, 128, 128)
        fire3 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire3.get_name(), fire3, output_channle)

        planes = (32, 128, 128)
        fire4 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire4.get_name(), fire4, output_channle)

        layer4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.addBlockList(BlockType.MyMaxPool2d, layer4, output_channle)

        planes = (48, 192, 192)
        fire5 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire5.get_name(), fire5, output_channle)

        planes = (48, 192, 192)
        fire6 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire6.get_name(), fire6, output_channle)

        planes = (64, 256, 256)
        fire7 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire7.get_name(), fire7, output_channle)

        planes = (64, 256, 256)
        fire8 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire8.get_name(), fire8, output_channle)

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


class DilatedSqueezeNet(BaseModel):
    def __init__(self, data_channel=3, bnName=BatchNormType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BaseModelName.DilatedSqueezeNet)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 64

        self.outChannelList = []
        self.index = 0

        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvActivationBlock(in_channels=self.data_channel,
                                     out_channels=self.first_output,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     dilation=1,
                                     activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.addBlockList(BlockType.MyMaxPool2d, layer2, self.first_output)

        planes = (16, 64, 64)
        fire1 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire1.get_name(), fire1, output_channle)

        planes = (16, 64, 64)
        fire2 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire2.get_name(), fire2, output_channle)

        layer3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.addBlockList(BlockType.MyMaxPool2d, layer3, output_channle)

        planes = (32, 128, 128)
        fire3 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire3.get_name(), fire3, output_channle)

        planes = (32, 128, 128)
        fire4 = FireBlock(self.outChannelList[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire4.get_name(), fire4, output_channle)

        layer4 = nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=False)
        self.addBlockList(BlockType.MyMaxPool2d, layer4, output_channle)

        planes = (48, 192, 192)
        fire5 = FireBlock(self.outChannelList[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire5.get_name(), fire5, output_channle)

        planes = (48, 192, 192)
        fire6 = FireBlock(self.outChannelList[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire6.get_name(), fire6, output_channle)

        planes = (64, 256, 256)
        fire7 = FireBlock(self.outChannelList[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire7.get_name(), fire7, output_channle)

        planes = (64, 256, 256)
        fire8 = FireBlock(self.outChannelList[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.addBlockList(fire8.get_name(), fire8, output_channle)

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