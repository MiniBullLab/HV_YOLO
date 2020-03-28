#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.attention_block import SEBlock


class MobileNetBlockName():

    InvertedResidual = "invertedResidual"


class InvertedResidual(BaseBlock):
    def __init__(self, inp, hidden_dim, oup,
                 kernel_size, stride, use_se,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(MobileNetBlockName.InvertedResidual)
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                ConvBNActivationBlock(in_channels=hidden_dim,
                                      out_channels=hidden_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=(kernel_size - 1) // 2,
                                      groups=hidden_dim,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                # Squeeze-and-Excite
                SEBlock(hidden_dim, reduction=4) if use_se else nn.Sequential(),
                # pw-linear
                ConvBNActivationBlock(in_channels=hidden_dim,
                                      out_channels=oup,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=ActivationType.Linear)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                ConvBNActivationBlock(in_channels=inp,
                                      out_channels=hidden_dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                # dw
                ConvBNActivationBlock(in_channels=hidden_dim,
                                      out_channels=hidden_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=(kernel_size - 1) // 2,
                                      groups=hidden_dim,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=ActivationType.Linear),
                # Squeeze-and-Excite
                SEBlock(hidden_dim, reduction=4) if use_se else nn.Sequential(),
                ActivationLayer(activation_name=activation_name, inplace=False),
                # pw-linear
                ConvBNActivationBlock(in_channels=hidden_dim,
                                      out_channels=oup,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=ActivationType.Linear)
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
