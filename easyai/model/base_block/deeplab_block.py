#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BatchNormType, ActivationType
from easyai.model.base_block.base_block import *
from easyai.model.base_block.utility_layer import ActivationLayer, NormalizeLayer
from easyai.model.base_block.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility_block import SeparableConv2dBNActivation
from easyai.model.base_block.bisenet_block import GlobalAvgPooling


class DeepLabBlockName():

    ASPP = "aspp"


class ASPP(BaseBlock):
    def __init__(self, in_channels=2048, out_channels=256, output_stride=16,
                 bn_name=BatchNormType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(DeepLabBlockName.ASPP)
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError

        self.image_pooling = GlobalAvgPooling(in_channels, out_channels,
                                              bn_name=bn_name,
                                              activation_name=activation_name)

        self.aspp0 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.aspp1 = SeparableConv2dBNActivation(in_channels, out_channels, dilation=dilations[0],
                                                 relu_first=False, bn_name=bn_name,
                                                 activation_name=activation_name)
        self.aspp2 = SeparableConv2dBNActivation(in_channels, out_channels, dilation=dilations[1],
                                                 relu_first=False, bn_name=bn_name,
                                                 activation_name=activation_name)
        self.aspp3 = SeparableConv2dBNActivation(in_channels, out_channels, dilation=dilations[2],
                                                 relu_first=False, bn_name=bn_name,
                                                 activation_name=activation_name)

        self.conv = ConvBNActivationBlock(in_channels=out_channels*5,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          bias=False,
                                          bnName=bn_name,
                                          activationName=activation_name)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)

        x = self.conv(x)
        x = self.dropout(x)

        return x
