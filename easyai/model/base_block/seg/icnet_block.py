#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.upsample_layer import Upsample


class ICNetBlockName():

    InputDownSample = "InputDownSample"
    CascadeFeatureFusion = "CascadeFeatureFusion"
    PyramidPoolingBlock = "PyramidPoolingBlock"


class InputDownSample(BaseBlock):

    def __init__(self,  mode='bilinear'):
        super().__init__(ICNetBlockName.InputDownSample)
        self.down1 = Upsample(scale_factor=0.5, mode=mode)
        self.down2 = Upsample(scale_factor=0.25, mode=mode)

    def forward(self, x):
        x_sub2 = self.down1(x)
        x_sub4 = self.down2(x)
        result = (x_sub2, x_sub4)
        return result


class CascadeFeatureFusion(BaseBlock):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, scale,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(ICNetBlockName.CascadeFeatureFusion)
        self.up = Upsample(scale_factor=scale, mode='bilinear')
        self.conv_low = ConvBNActivationBlock(in_channels=low_channels,
                                              out_channels=out_channels,
                                              kernel_size=3,
                                              padding=2,
                                              dilation=2,
                                              bias=False,
                                              bnName=bn_name,
                                              activationName=ActivationType.Linear)
        self.conv_high = ConvBNActivationBlock(in_channels=high_channels,
                                               out_channels=out_channels,
                                               kernel_size=1,
                                               bias=False,
                                               bnName=bn_name,
                                               activationName=ActivationType.Linear)
        self.activation = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x_low, x_high):
        x_low = self.up(x_low)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = self.activation(x)
        return x


class PyramidPoolingBlock(BaseBlock):

    def __init__(self, pyramids=(1, 2, 3, 6)):
        super().__init__(ICNetBlockName.PyramidPoolingBlock)
        self.pyramids = pyramids

    def forward(self, input_data):
        feat = input_data
        height, width = input_data.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input_data, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat = feat + x
        return feat
