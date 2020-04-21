#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.upsample_layer import Upsample


class BiSeNetBlockName():

    SpatialPath = "spatialPath"
    GlobalAvgPooling = "globalAvgPooling"
    AttentionRefinmentBlock = "attentionRefinmentBlock"
    ContextPath = "contextPath"
    FeatureFusionBlock = "featureFusionBlock"


class SpatialPath(BaseBlock):
    """Spatial path"""

    def __init__(self, in_channels, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BiSeNetBlockName.SpatialPath)
        inter_channels = 64
        self.conv7x7 = ConvBNActivationBlock(in_channels=in_channels,
                                             out_channels=inter_channels,
                                             kernel_size=7,
                                             stride=2,
                                             padding=3,
                                             bnName=bn_name,
                                             activationName=activation_name)
        self.conv3x3_1 = ConvBNActivationBlock(in_channels=inter_channels,
                                               out_channels=inter_channels,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               bnName=bn_name,
                                               activationName=activation_name)
        self.conv3x3_2 = ConvBNActivationBlock(in_channels=inter_channels,
                                               out_channels=inter_channels,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               bnName=bn_name,
                                               activationName=activation_name)
        self.conv1x1 = ConvBNActivationBlock(in_channels=inter_channels,
                                             out_channels=out_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bnName=bn_name,
                                             activationName=activation_name)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        return x


class GlobalAvgPooling(BaseBlock):
    def __init__(self, in_channels, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BiSeNetBlockName.GlobalAvgPooling)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNActivationBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  bias=False,
                                  bnName=bn_name,
                                  activationName=activation_name)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class AttentionRefinmentBlock(BaseBlock):
    def __init__(self, in_channels, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BiSeNetBlockName.AttentionRefinmentBlock)
        self.conv3x3 = ConvBNActivationBlock(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bnName=bn_name,
                                             activationName=activation_name)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNActivationBlock(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bnName=bn_name,
                                  activationName=activation_name),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


class ContextPath(BaseBlock):

    def __init__(self, in_channels, out_channels, scale_factor=2,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BiSeNetBlockName.ContextPath)

        self.arm = AttentionRefinmentBlock(in_channels, out_channels,
                                           bn_name=bn_name, activation_name=activation_name)
        self.refine = ConvBNActivationBlock(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bnName=bn_name,
                                            activationName=activation_name)
        self.up = Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, last_feature, x1):
        feature = self.arm(x1)
        feature += last_feature
        last_feature = self.up(feature)
        last_feature = self.refine(last_feature)
        return last_feature


class FeatureFusionBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, reduction=1,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BiSeNetBlockName.FeatureFusionBlock)
        self.conv1x1 = ConvBNActivationBlock(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bnName=bn_name,
                                             activationName=activation_name)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNActivationBlock(in_channels=out_channels,
                                  out_channels=out_channels // reduction,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(in_channels=out_channels // reduction,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bnName=bn_name,
                                  activationName=activation_name),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out
