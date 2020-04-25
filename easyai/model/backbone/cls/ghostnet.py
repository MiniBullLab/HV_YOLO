#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
title={GhostNet: More Features from Cheap Operations},
author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
booktitle={CVPR},
year={2020}
https://arxiv.org/abs/1911.11907
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.ghostnet_block import GhostBottleneck


__all__ = ['ghost_net']


class GhostNet(BaseBackbone):
    def __init__(self, cfgs, data_channel=3, width_mult=1.,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.GhostNet)
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.data_channel = data_channel
        self.width_mult = width_mult
        self.activation_name = activationName
        self.bn_name = bnName

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()
        # building first layer
        output_channel = self.make_divisible(16 * self.width_mult, 4)
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=output_channel,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        input_channel = output_channel
        output_channel = self.make_layer(input_channel, GhostBottleneck, self.cfgs)

        # building last several layers
        input_channel = self.block_out_channels[-1]
        layer2 = ConvBNActivationBlock(in_channels=input_channel,
                                       out_channels=output_channel,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, output_channel)

    def make_layer(self, input_channel, block, cfgs):
        # building inverted residual blocks
        hidden_channel = 0
        for k, exp_size, c, use_se, s in cfgs:
            output_channel = self.make_divisible(c * self.width_mult, 4)
            hidden_channel = self.make_divisible(exp_size * self.width_mult, 4)
            temp_block = block(input_channel, hidden_channel, output_channel, k, s, use_se)
            self.add_block_list(temp_block.get_name(), temp_block, output_channel)
            input_channel = output_channel
        return hidden_channel

    def make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def ghost_net(data_channel):
    cfgs = [
        # k, t, c, SE, s
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [5, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    model = GhostNet(cfgs=cfgs, data_channel=data_channel)
    model.set_name(BackboneName.GhostNet)
    return model
