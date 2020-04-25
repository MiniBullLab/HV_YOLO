#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.separable_conv_block import SeparableConv2dBNActivation
from easyai.model.base_block.seg.bisenet_block import GlobalAvgPooling


class DeepLabBlockName():

    ASPP = "aspp"


class ASPP(BaseBlock):
    def __init__(self, in_channels=2048, out_channels=256, output_stride=16,
                 bn_name=NormalizationType.BatchNormalize2d,
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


class ASPPBlock(BaseBlock):

    def __init__(self, features, inner_features=512, out_features=512, dilations=(12, 24, 36),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(DeepLabBlockName.ASPP)
        self.image_pooling = GlobalAvgPooling(features, inner_features,
                                              bn_name=bn_name,
                                              activation_name=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=features,
                                           out_channels=inner_features,
                                           kernel_size=1,
                                           padding=0,
                                           dilation=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv3 = ConvBNActivationBlock(in_channels=features,
                                           out_channels=inner_features,
                                           kernel_size=3,
                                           padding=dilations[0],
                                           dilation=dilations[0],
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv4 = ConvBNActivationBlock(in_channels=features,
                                           out_channels=inner_features,
                                           kernel_size=3,
                                           padding=dilations[1],
                                           dilation=dilations[1],
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv5 = ConvBNActivationBlock(in_channels=features,
                                           out_channels=inner_features,
                                           kernel_size=3,
                                           padding=dilations[2],
                                           dilation=dilations[2],
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.bottleneck = nn.Sequential(
            ConvBNActivationBlock(in_channels=inner_features * 5,
                                  out_channels=out_features,
                                  kernel_size=1,
                                  bias=False,
                                  bnName=bn_name,
                                  activationName=activation_name),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):

        feat1 = self.image_pooling(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle
