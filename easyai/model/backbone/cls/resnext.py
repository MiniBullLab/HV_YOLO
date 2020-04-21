#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""resnext in pytorch
[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.

    Aggregated Residual Transformations for Deep Neural Networks
    https://arxiv.org/abs/1611.05431
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.resnext_block import ResNextBottleNeck

__all__ = ['resnext50', 'resnext101', 'resnext152']


class ResNext(BaseBackbone):

    def __init__(self, data_channel=3,
                 block=ResNextBottleNeck, num_blocks=(3, 4, 6, 3),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.ResNext50)
        self.data_channel = data_channel
        self.block = block
        self.num_blocks = num_blocks
        self.out_channels = (64, 128, 256, 512)
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
                                       stride=1,
                                       padding=1,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(block1.get_name(), block1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_layer(self.block, num_block, self.out_channels[index],
                            self.strides[index])
            self.in_channel = self.out_channels[index] * 4

    def make_layer(self, block, num_block, out_channel, stride):
        strides = [stride] + [1] * (num_block - 1)
        temp_output_channel = out_channel * 4
        for stride in strides:
            temp_block = block(self.in_channel, out_channel, stride,
                               bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, temp_output_channel)
            self.in_channel = temp_output_channel

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def resnext50(data_channel):
    """ return a resnext50(c32x4d) network
    """
    model = ResNext(data_channel=data_channel,
                    num_blocks=(3, 4, 6, 3))
    model.set_name(BackboneName.ResNext50)
    return model


def resnext101(data_channel):
    """ return a resnext101(c32x4d) network
    """
    model = ResNext(data_channel=data_channel,
                    num_blocks=(3, 4, 23, 3))
    model.set_name(BackboneName.ResNext101)
    return model


def resnext152(data_channel):
    """ return a resnext152(c32x4d) network
    """
    model = ResNext(data_channel=data_channel,
                    num_blocks=(3, 4, 36, 3))
    model.set_name(BackboneName.ResNext152)
    return model



