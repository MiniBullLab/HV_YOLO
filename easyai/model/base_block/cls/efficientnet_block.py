#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.attention_block import SEConvBlock


class EfficientNetBlockName():

    EfficientNetBlock = "efficientNetBlock"


class EfficientNetBlock(BaseBlock):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, expansion, stride,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(EfficientNetBlockName.EfficientNetBlock)
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = ConvBNActivationBlock(in_channels=in_planes,
                                           out_channels=planes,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=planes,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           groups=planes,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv3 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=out_planes,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=ActivationType.Linear)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = ConvBNActivationBlock(in_channels=in_planes,
                                                  out_channels=out_planes,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=False,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)

        # SE layers
        self.se_block = SEConvBlock(out_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = self.se_block(out)
        out = out * w + shortcut
        return out
