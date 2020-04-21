#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

"""sknet in pytorch
title={Selective Kernel Networks},
author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},
journal={IEEE Conference on Computer Vision and Pattern Recognition},
year={2019}
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.sknet_block import SKBlock

__all__ = ['sknet50', 'sknet101']


class SKNet(BaseBackbone):

    def __init__(self, data_channel=3,
                 block=SKBlock, num_blocks=(3, 4, 6, 3),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.SKNet50)
        self.data_channel = data_channel
        self.block = block
        self.num_blocks = num_blocks
        self.out_channels = (128, 256, 512, 1024)
        self.strides = (1, 2, 2, 2)
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 64
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        conv1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                      out_channels=self.first_output,
                                      kernel_size=7,
                                      stride=2,
                                      padding=3,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, self.first_output)

        maxpool = nn.MaxPool2d(3, 2, 1)
        self.add_block_list(LayerType.MyMaxPool2d, maxpool, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_layer(self.block, num_block, self.out_channels[index],
                            self.strides[index])

    def make_layer(self, block, nums_block, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != planes * block.expansion:
            downsample = ConvBNActivationBlock(in_channels=self.in_channel,
                                               out_channels=planes * block.expansion,
                                               kernel_size=1,
                                               stride=stride,
                                               bias=False,
                                               bnName=self.bn_name,
                                               activationName=ActivationType.Linear)
        down_block = block(self.in_channel, planes, stride, downsample,
                           bn_name=self.bn_name, activation_name=self.activation_name)
        out_channels = planes * block.expansion
        self.add_block_list(down_block.get_name(), down_block, out_channels)

        self.in_channel = out_channels
        for _ in range(1, nums_block):
            temp_block = block(self.in_channel, planes,
                               bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, out_channels)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            print(key, x.shape)
        return output_list


def sknet50(data_channel):
    model = SKNet(data_channel=data_channel,
                  num_blocks=(3, 4, 6, 3))
    model.set_name(BackboneName.SKNet50)
    return model


def sknet101(data_channel):
    model = SKNet(data_channel=data_channel,
                  num_blocks=(3, 4, 23, 3))
    model.set_name(BackboneName.SKNet101)
    return model
