#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.residual_block import ResidualBlock


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResNet(BaseBackbone):
    def __init__(self, data_channel=3, num_blocks=[2, 2, 2, 2], out_channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1], bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU, block_flag=0):
        super().__init__()
        self.set_name(BackboneName.ResNet18)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.block_flag = block_flag
        self.first_output = 64
        self.in_channels = self.first_output
        self.head_type = 1

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        if self.head_type == 0:
            layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                           out_channels=self.first_output,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bnName=self.bnName,
                                           activationName=self.activationName)
            self.add_block_list(layer1.get_name(), layer1, self.first_output)
        elif self.head_type == 1:
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
        elif self.head_type == 2:
            layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                           out_channels=self.first_output,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           bnName=self.bnName,
                                           activationName=self.activationName)
            self.add_block_list(layer1.get_name(), layer1, self.first_output)

            layer11 = ConvBNActivationBlock(in_channels=self.first_output,
                                            out_channels=self.first_output,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bnName=self.bnName,
                                            activationName=self.activationName)
            self.add_block_list(layer11.get_name(), layer11, self.first_output)

            layer12 = ConvBNActivationBlock(in_channels=self.first_output,
                                            out_channels=self.first_output,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bnName=self.bnName,
                                            activationName=self.activationName)
            self.add_block_list(layer12.get_name(), layer12, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_resnet_block(self.out_channels[index], self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.bnName, self.activationName,
                                   self.block_flag)
            self.in_channels = self.block_out_channels[-1]

    def make_resnet_block(self, out_channels, num_block, stride, dilation,
                         bn_name, activation, block_flag):
        expansion = 0
        if block_flag == 0:
            expansion = 1
        elif block_flag == 1:
            expansion = 4
        down_layers = ResidualBlock(self.block_flag, self.in_channels, out_channels, stride,
                                    dilation=dilation, expansion=expansion, bn_name=bn_name,
                                    activation_name=activation)
        name = "down_%s" % down_layers.get_name()
        temp_output_channel = out_channels * expansion
        self.add_block_list(name, down_layers, temp_output_channel)
        for i in range(num_block - 1):
            layer = ResidualBlock(self.block_flag, temp_output_channel, out_channels, expansion=expansion,
                                  bn_name=bn_name, activation_name=activation)
            temp_output_channel = out_channels * expansion
            self.add_block_list(layer.get_name(), layer, temp_output_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def resnet18(data_channel):
    model = ResNet(data_channel=data_channel,
                   num_blocks=[2, 2, 2, 2],
                   block_flag=0)
    model.set_name(BackboneName.ResNet18)
    return model


def resnet34(data_channel):
    model = ResNet(data_channel=data_channel,
                   num_blocks=[3, 4, 6, 3],
                   block_flag=0)
    model.set_name(BackboneName.ResNet34)
    return model


def resnet50(data_channel):
    model = ResNet(data_channel=data_channel,
                   num_blocks=[3, 4, 6, 3],
                   block_flag=1)
    model.set_name(BackboneName.ResNet50)
    return model


def resnet101(data_channel):
    model = ResNet(data_channel=data_channel,
                   num_blocks=[3, 4, 23, 3],
                   block_flag=1)
    model.set_name(BackboneName.ResNet101)
    return model


def resnet152(data_channel):
    model = ResNet(data_channel=data_channel,
                   num_blocks=[3, 8, 36, 3],
                   block_flag=1)
    model.set_name(BackboneName.ResNet152)
    return model
