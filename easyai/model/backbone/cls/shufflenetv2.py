#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
# ShuffleNetV2 in PyTorch.
# See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import BatchNormType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility_layer import MyMaxPool2d
from easyai.model.base_block.utility_block import ConvBNActivationBlock
from easyai.model.base_block.shufflenet_block import DownBlock, BasicBlock


__all__ = ['shufflenetv2_1_0']


class ShuffleNetV2(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=[3, 7, 3], out_channels=(116, 232, 464),
                 strides=[2, 2, 2], dilations=[1, 1, 1], bnName=BatchNormType.BatchNormalize2d,
                 activationName=ActivationType.LeakyReLU):
        super().__init__()
        # init param
        self.set_name(BackboneName.ShuffleNetV2_1_0)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 24

        self.create_block_list()

    def create_block_list(self):
        self.out_channels = []
        self.index = 0

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = MyMaxPool2d(kernel_size=3, stride=2)
        self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_suffle_block(self.out_channels[index], self.num_blocks[index],
                                 self.strides[index], self.dilations[index],
                                 self.bnName, self.activationName)
            self.in_channels = self.outChannelList[-1]

    def make_suffle_block(self, out_channels, num_blocks, stride, dilation,
                          bnName, activationName):
        downLayer = DownBlock(self.in_channels, out_channels, stride=stride,
                            bnName=bnName, activationName=activationName)
        self.add_block_list(downLayer.get_name(), downLayer, out_channels)
        for _ in range(num_blocks):
            tempLayer = BasicBlock(out_channels, out_channels, stride=1, dilation=dilation,
                 bnName=bnName, activationName=activationName)
            self.add_block_list(tempLayer.get_name(), tempLayer, out_channels)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def shufflenetv2_1_0():
    model = ShuffleNetV2(num_blocks=[3, 7, 3])
    model.set_name(BackboneName.ShuffleNetV2_1_0)
    return model


def shufflenet_v2_x0_5():
    model = ShuffleNetV2(num_blocks=[3, 7, 3], out_channels=(48, 96, 192))
    model.set_name(BackboneName.ShuffleNetV2_1_0)
    return model


def shufflenet_v2_x1_0():
    model = ShuffleNetV2(num_blocks=[3, 7, 3], out_channels=(116, 232, 464))
    model.set_name(BackboneName.ShuffleNetV2_1_0)
    return model


def shufflenet_v2_x1_5():
    model = ShuffleNetV2(num_blocks=[3, 7, 3], out_channels=(176, 352, 704))
    model.set_name(BackboneName.ShuffleNetV2_1_0)
    return model


def shufflenet_v2_x2_0():
    model = ShuffleNetV2(num_blocks=[3, 7, 3], out_channels=(244, 488, 976))
    model.set_name(BackboneName.ShuffleNetV2_1_0)
    return model
