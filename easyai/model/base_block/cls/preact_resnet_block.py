#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import BNActivationConvBlock


class PreActResNetBlockName():

    PreActBasic = "preActBasic"
    PreActBottleNeck = "preActBottleNeck"


class PreActBasic(BaseBlock):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PreActResNetBlockName.PreActBasic)
        self.residual = nn.Sequential(
            BNActivationConvBlock(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  bnName=bn_name,
                                  activationName=activation_name),
            BNActivationConvBlock(in_channels=out_channel,
                                  out_channels=out_channel * PreActBasic.expansion,
                                  kernel_size=3,
                                  padding=1,
                                  bnName=bn_name,
                                  activationName=activation_name)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channel, out_channel * PreActBasic.expansion,
                                      1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class PreActBottleNeck(BaseBlock):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PreActResNetBlockName.PreActBottleNeck)

        self.residual = nn.Sequential(
            BNActivationConvBlock(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=1,
                                  stride=stride,
                                  bnName=bn_name,
                                  activationName=activation_name),
            BNActivationConvBlock(in_channels=out_channel,
                                  out_channels=out_channel,
                                  kernel_size=3,
                                  padding=1,
                                  bnName=bn_name,
                                  activationName=activation_name),
            BNActivationConvBlock(in_channels=out_channel,
                                  out_channels=out_channel * PreActBottleNeck.expansion,
                                  kernel_size=1,
                                  bnName=bn_name,
                                  activationName=activation_name)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channel, out_channel * PreActBottleNeck.expansion,
                                      1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut
