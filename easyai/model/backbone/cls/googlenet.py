#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BatchNormType, ActivationType, BlockType
from easyai.base_name.base_model_name import BaseModelName
from easyai.model.backbone.utility.base_model import *
from easyai.model.base_block.utility_block import ConvBNActivationBlock
from easyai.model.base_block.googlenet_block import InceptionBlock


__all__ = ['GoogleNet']


class GoogleNet(BaseModel):

    def __init__(self, data_channel=3, bnName=BatchNormType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BaseModelName.GoogleNet)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 192

        self.outChannelList = []
        self.index = 0

        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.first_output)

        self.input_channel = self.first_output
        planes = (64, 96, 128, 16, 32, 32)
        layer2 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer2.get_name(), layer2, output_channel)

        self.input_channel = self.outChannelList[-1]
        planes = (128, 128, 192, 32, 96, 64)
        layer3 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer3.get_name(), layer3, output_channel)

        maxpool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.addBlockList(BlockType.MyMaxPool2d, maxpool1, self.outChannelList[-1])

        self.input_channel = self.outChannelList[-1]
        planes = (192, 96, 208, 16, 48, 64)
        layer4 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer4.get_name(), layer4, output_channel)

        self.input_channel = self.outChannelList[-1]
        planes = (160, 112, 224, 24, 64, 64)
        layer5 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer5.get_name(), layer5, output_channel)

        self.input_channel = self.outChannelList[-1]
        planes = (128, 128, 256, 24, 64, 64)
        layer6 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer6.get_name(), layer6, output_channel)

        self.input_channel = self.outChannelList[-1]
        planes = (112, 144, 288, 32, 64, 64)
        layer7 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer7.get_name(), layer7, output_channel)

        self.input_channel = self.outChannelList[-1]
        planes = (256, 160, 320, 32, 128, 128)
        layer8 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer8.get_name(), layer8, output_channel)

        maxpool2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.addBlockList(BlockType.MyMaxPool2d, maxpool2, self.outChannelList[-1])

        self.input_channel = self.outChannelList[-1]
        planes = (256, 160, 320, 32, 128, 128)
        layer9 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer9.get_name(), layer9, output_channel)

        self.input_channel = self.outChannelList[-1]
        planes = (384, 192, 384, 48, 128, 128)
        layer10 = InceptionBlock(self.input_channel, planes, bnName=self.bnName,
                                activationName=self.activationName)
        output_channel = planes[0] + planes[2] + planes[4] + planes[5]
        self.addBlockList(layer10.get_name(), layer10, output_channel)

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
