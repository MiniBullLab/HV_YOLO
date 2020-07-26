#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.separable_conv_block import DepthwiseConv2dBlock
from easyai.model.base_block.utility.attention_block import SEBlock


class EfficientNetBlockName():

    MBConvBlock = "MBConvBlock"


class MBConvBlock(BaseBlock):
    """ Inverted residual block """

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, expand_ratio=1, use_se=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.Swish):
        super().__init__(EfficientNetBlockName.MBConvBlock)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        self.stride = stride

        self.expand = nn.Sequential()
        if expand_ratio != 1:
            self.expand = ConvBNActivationBlock(in_channels=in_channel,
                                                out_channels=in_channel*expand_ratio,
                                                kernel_size=1,
                                                bias=False,
                                                bnName=bn_name,
                                                activationName=activation_name)

        self.dw_conv = DepthwiseConv2dBlock(in_channel*expand_ratio, kernel_size,
                                            padding=(kernel_size-1)//2, stride=stride,
                                            bn_name=bn_name,
                                            activation_name=activation_name)
        self.se_block = nn.Sequential()
        if use_se:
            self.se_block = SEBlock(in_channel*expand_ratio, reduction=4,
                                    activate_name=activation_name)

        self.pw_conv = ConvBNActivationBlock(in_channels=in_channel*expand_ratio,
                                             out_channels=out_channel,
                                             kernel_size=1,
                                             bias=False,
                                             bnName=bn_name,
                                             activationName=ActivationType.Linear)

    def forward(self, input_data):
        x = self.expand(input_data)
        x = self.dw_conv(x)
        x = self.se_block(x)
        x = self.pw_conv(x)
        if self.in_channel == self.out_channel and \
                self.stride == 1:
            x = x + input_data
        return x
