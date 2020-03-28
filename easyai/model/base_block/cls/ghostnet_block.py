#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import math
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.attention_block import SEBlock


class GhostNetBlockName():

    GhostBlock = "ghostBlock"
    GhostBottleneck = "ghostBottleneck"


class GhostBlock(BaseBlock):
    def __init__(self, inp, oup, kernel_size=1,
                 ratio=2, dw_size=3, stride=1,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(GhostNetBlockName.GhostBlock)
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvBNActivationBlock(in_channels=inp,
                                                  out_channels=init_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=kernel_size // 2,
                                                  bias=False,
                                                  bnName=bn_name,
                                                  activationName=activation_name)

        self.cheap_operation = ConvBNActivationBlock(in_channels=init_channels,
                                                     out_channels=new_channels,
                                                     kernel_size=dw_size,
                                                     stride=1,
                                                     padding=dw_size // 2,
                                                     groups=init_channels,
                                                     bias=False,
                                                     bnName=bn_name,
                                                     activationName=activation_name)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(BaseBlock):
    def __init__(self, inp, hidden_dim, oup,
                 kernel_size, stride, use_se,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(GhostNetBlockName.GhostBottleneck)
        assert stride in [1, 2]

        if stride == 2:
            depthwise_conv = ConvBNActivationBlock(in_channels=hidden_dim,
                                                   out_channels=hidden_dim,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=kernel_size // 2,
                                                   groups=hidden_dim,
                                                   bias=False,
                                                   bnName=bn_name,
                                                   activationName=ActivationType.Linear)
        else:
            depthwise_conv = nn.Sequential()

        self.conv = nn.Sequential(
            # pw
            GhostBlock(inp, hidden_dim, kernel_size=1,
                       bn_name=bn_name, activation_name=activation_name),
            # dw
            depthwise_conv,
            # Squeeze-and-Excite
            SEBlock(hidden_dim, reduction=4) if use_se else nn.Sequential(),
            # pw-linear
            GhostBlock(hidden_dim, oup, kernel_size=1,
                       bn_name=bn_name, activation_name=ActivationType.Linear),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            depthwise_conv = ConvBNActivationBlock(in_channels=inp,
                                                   out_channels=inp,
                                                   kernel_size=3,
                                                   stride=stride,
                                                   padding=1,
                                                   groups=inp,
                                                   bias=False,
                                                   bnName=bn_name,
                                                   activationName=activation_name)
            self.shortcut = nn.Sequential(
                depthwise_conv,
                ConvBNActivationBlock(in_channels=inp,
                                      out_channels=oup,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=ActivationType.Linear)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)