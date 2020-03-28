#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.residual_block import InvertedResidual
from easyai.model.base_block.seg.pspnet_block import PyramidPooling


class FastSCNNBlockName():

    GlobalFeatureExtractor = "globalFeatureExtractor"
    FeatureFusionBlock = "featureFusionBlock"


class GlobalFeatureExtractor(BaseBlock):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128,
                 t=6, num_blocks=(3, 3, 3), bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(FastSCNNBlockName.GlobalFeatureExtractor)
        self.bn_name = bn_name
        self.activation_name = activation_name

        self.bottleneck1 = self.make_layer(InvertedResidual, in_channels, block_channels[0],
                                           num_blocks[0], t, 2)
        self.bottleneck2 = self.make_layer(InvertedResidual, block_channels[0], block_channels[1],
                                            num_blocks[1], t, 2)
        self.bottleneck3 = self.make_layer(InvertedResidual, block_channels[1], block_channels[2],
                                            num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], bn_name=self.bn_name,
                                  activation_name=self.activation_name)
        self.out = ConvBNActivationBlock(in_channels=block_channels[2] * 2,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         bnName=self.bn_name,
                                         activationName=self.activation_name)

    def make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, t, bnName=self.bn_name))
        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, t, bnName=self.bn_name))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        x = self.out(x)
        return x


class FeatureFusionBlock(BaseBlock):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels,
                 scale_factor=4, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(FastSCNNBlockName.FeatureFusionBlock)
        self.scale_factor = scale_factor
        self.dwconv = ConvBNActivationBlock(in_channels=lower_in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1,
                                            bnName=bn_name,
                                            activationName=activation_name)
        self.conv_lower_res = ConvBNActivationBlock(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    bnName=bn_name,
                                                    activationName=ActivationType.Linear)
        self.conv_higher_res = ConvBNActivationBlock(in_channels=highter_in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=1,
                                                     bnName=bn_name,
                                                     activationName=ActivationType.Linear)
        self.relu = ActivationLayer(activation_name, inplace=False)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor,
                                          mode='bilinear', align_corners=False)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)
