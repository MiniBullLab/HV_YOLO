#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""senet in pytorch
[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.senet_block import BasicResidualSEBlock
from easyai.model.base_block.cls.senet_block import BottleneckResidualSEBlock

__all__ = ['se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152']


class SEResNet(BaseBackbone):

    def __init__(self, data_channel=3,
                 block=BasicResidualSEBlock, num_blocks=(2, 2, 2, 2),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.SEResNet18)
        self.data_channel = data_channel
        self.block = block
        self.num_blocks = num_blocks
        self.out_channels = (64, 128, 256, 516)
        self.strides = (1, 2, 2, 2)
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 64
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        block1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(block1.get_name(), block1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_layer(self.block, num_block, self.out_channels[index],
                            self.strides[index])
            self.in_channel = self.out_channels[index] * self.block.expansion

    def make_layer(self, block, num_block, out_channel, stride):
        down_block = block(self.in_channel, out_channel, stride,
                           bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(down_block.get_name(), down_block,
                            out_channel * block.expansion)
        temp_output_channel = out_channel * block.expansion

        for _ in range(num_block - 1):
            temp_block = block(temp_output_channel, out_channel, 1,
                               bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, temp_output_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def se_resnet18(data_channel):
    model = SEResNet(data_channel=data_channel,
                     block=BasicResidualSEBlock,
                     num_blocks=(2, 2, 2, 2))
    model.set_name(BackboneName.SEResNet18)
    return model


def se_resnet34(data_channel):
    model = SEResNet(data_channel=data_channel,
                     block=BasicResidualSEBlock,
                     num_blocks=(3, 4, 6, 3))
    model.set_name(BackboneName.SEResNet34)
    return model


def se_resnet50(data_channel):
    model = SEResNet(data_channel=data_channel,
                     block=BottleneckResidualSEBlock,
                     num_blocks=(3, 4, 6, 3))
    model.set_name(BackboneName.SEResNet50)
    return model


def se_resnet101(data_channel):
    model = SEResNet(data_channel=data_channel,
                     block=BottleneckResidualSEBlock,
                     num_blocks=(3, 4, 23, 3))
    model.set_name(BackboneName.SEResNet101)
    return model


def se_resnet152(data_channel):
    model = SEResNet(data_channel=data_channel,
                     block=BottleneckResidualSEBlock,
                     num_blocks=(3, 8, 36, 3))
    model.set_name(BackboneName.SEResNet152)
    return model
