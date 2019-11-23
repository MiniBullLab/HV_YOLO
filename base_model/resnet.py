#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_model.base_model import *
from base_name.base_model_name import BaseModelName
from base_name.block_name import BatchNormType, ActivationType, BlockType
from base_block.utility_block import ConvBNActivationBlock
from base_block.resnet_block import BasicBlock, Bottleneck


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResNet(BaseModel):
    def __init__(self, data_channel=3, num_blocks=[2, 2, 2, 2], out_channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1], bnName=BatchNormType.BatchNormalize,
                 activationName=ActivationType.ReLU, block=BasicBlock):
        super().__init__()
        self.set_name(BaseModelName.ResNet18)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.block = block
        self.first_output = 64

        self.outChannelList = []
        self.index = 0
        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=7,
                                       stride=2,
                                       padding=3,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.first_output)

        # layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
        #                                out_channels=self.first_output,
        #                                kernel_size=3,
        #                                stride=2,
        #                                padding=1,
        #                                bnName=self.bnName,
        #                                activationName=self.activationName)
        # self.addBlockList(layer1.get_name(), layer1, self.first_output)
        # self.in_channels = self.first_output
        # self.first_output = 64
        # layer11 = ConvBNActivationBlock(in_channels=self.data_channel,
        #                                out_channels=self.first_output,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1,
        #                                bnName=self.bnName,
        #                                activationName=self.activationName)
        # self.addBlockList(layer11.get_name(), layer11, self.first_output)
        # self.in_channels = self.first_output
        # self.first_output = 128
        # layer12 = ConvBNActivationBlock(in_channels=self.data_channel,
        #                                out_channels=self.first_output,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1,
        #                                bnName=self.bnName,
        #                                activationName=self.activationName)
        # self.addBlockList(layer12.get_name(), layer12, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.addBlockList(BlockType.MyMaxPool2d, layer2, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_resnet_layer(self.out_channels[index], self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.bnName, self.activationName,
                                   self.block)
            self.in_channels = self.outChannelList[-1]

    def make_resnet_layer(self, out_channels, num_block, stride, dilation,
                          bnName, activation, block):
        down_layers = block(self.in_channels, out_channels, stride,
                            dilation=dilation, bnName=bnName,
                            activationName=activation)
        name = "down_%s" % down_layers.get_name()
        temp_output_channel = out_channels * block.expansion
        self.addBlockList(name, down_layers, temp_output_channel)
        for i in range(num_block - 1):
            layer = block(temp_output_channel, out_channels, 1,
                          bnName=bnName, activationName=activation)
            temp_output_channel = out_channels * block.expansion
            self.addBlockList(layer.get_name(), layer, temp_output_channel)

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


def resnet18():
    model = ResNet(num_blocks=[2, 2, 2, 2], block=BasicBlock)
    model.set_name(BaseModelName.ResNet18)
    return model


def resnet34():
    model = ResNet(num_blocks=[3, 4, 6, 3], block=BasicBlock)
    model.set_name(BaseModelName.ResNet34)
    return model


def resnet50():
    model = ResNet(num_blocks=[3, 4, 6, 3], block=Bottleneck)
    model.set_name(BaseModelName.ResNet50)
    return model


def resnet101():
    model = ResNet(num_blocks=[3, 4, 23, 3], block=Bottleneck)
    model.set_name(BaseModelName.ResNet101)
    return model


def resnet152():
    model = ResNet(num_blocks=[3, 8, 36, 3], block=Bottleneck)
    model.set_name(BaseModelName.ResNet152)
    return model
