#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
# ShuffleNetV2 in PyTorch.
# See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.

from base_model.base_model import *
from base_name.base_model_name import BaseModelName
from base_name.block_name import BatchNormType, ActivationType, BlockType
from base_block.utility_layer import MyMaxPool2d
from base_block.utility_block import ConvBNActivationBlock
from base_block.shufflenet_block import DownBlock, BasicBlock


__all__ = ['shufflenet_v2_1_0']


class ShuffleNetV2(BaseModel):

    def __init__(self, data_channel=3, num_blocks=[3, 7, 3], out_channels=(24, 24, 116, 232, 464),
                 stride=[2, 2, 2], dilation=[1, 1, 1], bnName=BatchNormType.BatchNormalize,
                 activationName=ActivationType.ReLU):
        super().__init__()
        # init param
        self.set_name(BaseModelName.ShuffleNetV2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activationName = activationName
        self.bnName = bnName
        self.outChannelList = []
        self.index = 0

        self.createBlockList()

    def createBlockList(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.out_channels[0],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.out_channels[0])

        layer2 = MyMaxPool2d(kernel_size=3, stride=2)
        self.addBlockList(BlockType.MyMaxPool2d, layer2, self.out_channels[0])

        self.in_channels = self.out_channels[1]
        self.makeSuffleBlock(self.out_channels[2], self.num_blocks[0], self.stride[0], self.dilation[0],
                             self.bnName, self.activationName)
        self.in_channels = self.out_channels[2]
        self.makeSuffleBlock(self.out_channels[3], self.num_blocks[1], self.stride[0], self.dilation[1],
                             self.bnName, self.activationName)
        self.in_channels = self.out_channels[3]
        self.makeSuffleBlock(self.out_channels[4], self.num_blocks[2], self.stride[0], self.dilation[2],
                             self.bnName, self.activationName)

    def makeSuffleBlock(self, out_channels, num_blocks, stride, dilation, bnName, activationName):
        downLayer = DownBlock(self.in_channels, out_channels, stride=stride,
                            bnName=bnName, activationName=activationName)
        self.addBlockList(downLayer.get_name(), downLayer, out_channels)
        for _ in range(num_blocks):
            tempLayer = BasicBlock(out_channels, out_channels, stride=1, dilation=dilation,
                 bnName=bnName, activationName=activationName)
            self.addBlockList(tempLayer.get_name(), tempLayer, out_channels)

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


def shufflenet_v2_1_0():
    model = ShuffleNetV2()
    return model
