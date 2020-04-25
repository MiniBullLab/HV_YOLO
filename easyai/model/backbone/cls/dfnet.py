#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.dfnet_block import BasicBlock

__all__ = ['dfnetv1', 'dfnetv2']


class DFNet(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(3, 3, 3, 1),
                 out_channels=(64, 128, 256, 512), strides=(2, 2, 2, 1),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.DFNetV1)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 64
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        conv1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                      out_channels=32,
                                      kernel_size=3,
                                      padding=1,
                                      stride=2,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 32)

        conv2 = ConvBNActivationBlock(in_channels=32,
                                      out_channels=self.first_output,
                                      kernel_size=3,
                                      padding=1,
                                      stride=2,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self._make_layer(num_block, self.out_channels[index], self.strides[index])

    def _make_layer(self, num_block, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != planes * BasicBlock.expansion:
            downsample = ConvBNActivationBlock(in_channels=self.in_channel,
                                               out_channels=planes * BasicBlock.expansion,
                                               kernel_size=1,
                                               stride=stride,
                                               bias=False,
                                               bnName=self.bn_name,
                                               activationName=ActivationType.Linear)
        down_block = BasicBlock(self.in_channel, planes, stride, downsample,
                                bn_name=self.bn_name, activation_name=self.activation_name)
        down_name = "down_%s" % down_block.get_name()
        output_channle = planes * BasicBlock.expansion
        self.add_block_list(down_name, down_block, output_channle)

        self.in_channel = output_channle
        for i in range(1, num_block):
            temp_block = BasicBlock(self.in_channel, planes)
            self.add_block_list(temp_block.get_name(), temp_block, output_channle)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def dfnetv1(data_channel):
    model = DFNet(data_channel=data_channel,
                  num_blocks=(3, 3, 3, 1),
                  out_channels=(64, 128, 256, 512),
                  strides=(2, 2, 2, 1))
    model.set_name(BackboneName.DFNetV1)
    return model


def dfnetv2(data_channel):
    model = DFNet(data_channel=data_channel,
                  num_blocks=(2, 1, 10, 1, 4, 2),
                  out_channels=(64, 128, 128, 256, 256, 512),
                  strides=(2, 1, 2, 1, 2, 1))
    model.set_name(BackboneName.DFNetV2)
    return model

