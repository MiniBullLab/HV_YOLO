#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_model.base_model import *
from base_model.base_model_name import BaseModelName
from base_block.block_name import BatchNormType, ActivationType, BlockType
from base_block.utility_block import ConvBNActivationBlock
from base_block.resnet_block import BasicBlock, Bottleneck

__all__ = ['resnet_18']


class ResNet(BaseModel):
    def __init__(self, data_channel=3, num_blocks=[2,2,2,2], out_channels=[64,64,128,256,512],
                 stride=[1,2,2,2], dilation=[1,1,1,1], bnName=BatchNormType.BatchNormalize,
                 activationName=ActivationType.ReLU, block=BasicBlock):
        super().__init__()
        self.set_name(BaseModelName.ResNet)
        self.data_channel = data_channel
        self.num_blocks = num_blocks # [3, 7, 3]
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activationName = activationName
        self.bnName = bnName
        self.block = block

        self.outChannelList = []
        self.index = 0
        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.out_channels[0],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.out_channels[0])

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.addBlockList(BlockType.MyMaxPool2d, layer2, self.out_channels[0])

        self.in_channels = self.out_channels[1]
        self.make_resnet_layer(self.out_channels[1], self.num_blocks[0], self.stride[0], self.dilation[0],
                               self.bnName, self.activationName, self.block)

        self.in_channels = self.out_channels[1]
        self.make_resnet_layer(self.out_channels[2], self.num_blocks[1], self.stride[1], self.dilation[1],
                               self.bnName, self.activationName, self.block)

        self.in_channels = self.out_channels[2]
        self.make_resnet_layer(self.out_channels[3], self.num_blocks[2], self.stride[2], self.dilation[2],
                               self.bnName, self.activationName, self.block)

        self.in_channels = self.out_channels[3]
        self.make_resnet_layer(self.out_channels[4], self.num_blocks[3], self.stride[3], self.dilation[3],
                               self.bnName, self.activationName, self.block)

    def make_resnet_layer(self, out_channels, num_blocks, stride, dilation, BatchNorm, activation, block):
        down_layers = block(self.in_channels, out_channels, stride)
        name = "down_%s" % down_layers.get_name()
        self.addBlockList(name, down_layers, out_channels)
        for i in range(num_blocks - 1):
            layer = block(out_channels, out_channels, 1)
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


def get_resnet(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNet()

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def resnet_18(**kwargs):
    return get_resnet(**kwargs)