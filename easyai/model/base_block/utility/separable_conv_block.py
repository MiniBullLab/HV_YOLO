#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ActivationConvBNBlock
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class SeperableConv2dBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True):
        super().__init__(BlockType.SeperableConv2dBlock)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                         stride, padding, dilation, in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeparableConv2dBNActivation(BaseBlock):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.SeparableConv2dBNActivation)

        if relu_first:
            depthwise = ActivationConvBNBlock(in_channels=inplanes,
                                              out_channels=inplanes,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=dilation,
                                              dilation=dilation,
                                              groups=inplanes,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=activation_name)
            pointwise = ConvBNActivationBlock(in_channels=inplanes,
                                              out_channels=planes,
                                              kernel_size=1,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=ActivationType.Linear)
        else:
            depthwise = ConvBNActivationBlock(in_channels=inplanes,
                                              out_channels=inplanes,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=dilation,
                                              dilation=dilation,
                                              groups=inplanes,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=activation_name)
            pointwise = ConvBNActivationBlock(in_channels=inplanes,
                                              out_channels=planes,
                                              kernel_size=1,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=activation_name)

        self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                ('pointwise', pointwise)
                                                ]))

    def forward(self, x):
        return self.block(x)
