#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.resnet_block import BasicBlock, Bottleneck


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResNet(BaseBackbone):
    def __init__(self, data_channel=3, num_blocks=[2, 2, 2, 2], out_channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1], bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU, block=BasicBlock):
        super().__init__()
        self.set_name(BackboneName.ResNet18)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.block = block
        self.first_output = 64
        self.in_channels = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=7,
                                       stride=2,
                                       padding=3,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)

        # layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
        #                                out_channels=self.first_output,
        #                                kernel_size=3,
        #                                stride=2,
        #                                padding=1,
        #                                bnName=self.bnName,
        #                                activationName=self.activationName)
        # self.add_block_list(layer1.get_name(), layer1, self.first_output)
        # self.in_channels = self.first_output
        # self.first_output = 64
        # layer11 = ConvBNActivationBlock(in_channels=self.data_channel,
        #                                out_channels=self.first_output,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1,
        #                                bnName=self.bnName,
        #                                activationName=self.activationName)
        # self.add_block_list(layer11.get_name(), layer11, self.first_output)
        # self.in_channels = self.first_output
        # self.first_output = 128
        # layer12 = ConvBNActivationBlock(in_channels=self.data_channel,
        #                                out_channels=self.first_output,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1,
        #                                bnName=self.bnName,
        #                                activationName=self.activationName)
        # self.add_block_list(layer12.get_name(), layer12, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_resnet_layer(self.out_channels[index], self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.bnName, self.activationName,
                                   self.block)
            self.in_channels = self.block_out_channels[-1]

    def make_resnet_layer(self, out_channels, num_block, stride, dilation,
                          bnName, activation, block):
        down_layers = block(self.in_channels, out_channels, stride,
                            dilation=dilation, bnName=bnName,
                            activationName=activation)
        name = "down_%s" % down_layers.get_name()
        temp_output_channel = out_channels * block.expansion
        self.add_block_list(name, down_layers, temp_output_channel)
        for i in range(num_block - 1):
            layer = block(temp_output_channel, out_channels, 1,
                          bnName=bnName, activationName=activation)
            temp_output_channel = out_channels * block.expansion
            self.add_block_list(layer.get_name(), layer, temp_output_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def resnet18():
    model = ResNet(num_blocks=[2, 2, 2, 2], block=BasicBlock)
    model.set_name(BackboneName.ResNet18)
    return model


def resnet34():
    model = ResNet(num_blocks=[3, 4, 6, 3], block=BasicBlock)
    model.set_name(BackboneName.ResNet34)
    return model


def resnet50():
    model = ResNet(num_blocks=[3, 4, 6, 3], block=Bottleneck)
    model.set_name(BackboneName.ResNet50)
    return model


def resnet101():
    model = ResNet(num_blocks=[3, 4, 23, 3], block=Bottleneck)
    model.set_name(BackboneName.ResNet101)
    return model


def resnet152():
    model = ResNet(num_blocks=[3, 8, 36, 3], block=Bottleneck)
    model.set_name(BackboneName.ResNet152)
    return model
