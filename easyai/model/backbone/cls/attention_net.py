#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""residual attention network in pytorch
[1] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

    Residual Attention Network for Image Classification
    https://arxiv.org/abs/1704.06904
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.preact_resnet_block import PreActBottleNeck
from easyai.model.base_block.cls.attention_net_block import AttentionModule1
from easyai.model.base_block.cls.attention_net_block import AttentionModule2
from easyai.model.base_block.cls.attention_net_block import AttentionModule3

__all__ = ['attention_net56', 'attention_net92']


class AttentionNet(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(1, 1, 1),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.AttentionNet56)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = (256, 512, 1024)
        self.in_channels = (64, 256, 512)
        self.activation_name = activationName
        self.bn_name = bnName

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        block1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.in_channels[0],
                                       kernel_size=3,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(block1.get_name(), block1, self.in_channels[0])

        self.make_stage(self.in_channels[0], self.out_channels[0],
                        self.num_blocks[0], AttentionModule1)
        self.make_stage(self.in_channels[1], self.out_channels[1],
                        self.num_blocks[1], AttentionModule2)
        self.make_stage(self.in_channels[2], self.out_channels[2],
                        self.num_blocks[2], AttentionModule3)

        output_channle = 2048
        bottleneck_channels = int(output_channle / 4)
        block2 = PreActBottleNeck(self.out_channels[2], bottleneck_channels, 2)
        self.add_block_list(block2.get_name(), block2, output_channle)

        block3 = PreActBottleNeck(output_channle, bottleneck_channels, 1)
        self.add_block_list(block3.get_name(), block3, output_channle)

        block4 = PreActBottleNeck(output_channle, bottleneck_channels, 1)
        self.add_block_list(block4.get_name(), block4, output_channle)

    def make_stage(self, in_channel, out_channel, num, block):
        bottleneck_channels = int(out_channel / 4)
        down_block = PreActBottleNeck(in_channel, bottleneck_channels, 2)
        self.add_block_list(down_block.get_name(), down_block, out_channel)
        for _ in range(num):
            temp_block = block(out_channel, out_channel)
            self.add_block_list(temp_block.get_name(), temp_block, out_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def attention_net56(data_channel):
    model = AttentionNet(data_channel=data_channel,
                      num_blocks=(1, 1, 1))
    model.set_name(BackboneName.AttentionNet56)
    return model


def attention_net92(data_channel):
    model = AttentionNet(data_channel=data_channel,
                      num_blocks=(1, 2, 3))
    model.set_name(BackboneName.AttentionNet92)
    return model

