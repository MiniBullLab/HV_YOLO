#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.attention_block import SEConvBlock
from easyai.model.base_block.utility.utility_layer import ActivationLayer, NormalizeLayer


def split_channels(channels, num_groups):
    result_channels = [channels//num_groups for _ in range(num_groups)]
    result_channels[0] += channels - sum(result_channels)
    return result_channels


class MixNetBlockName():

    GroupedConv2d = "GroupedConv2d"
    MDConvBlock = "MDConvBlock"
    MixNetBlock = "MixNetBlock"


class GroupedConv2d(BaseBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(MixNetBlockName.GroupedConv2d)
        self.num_groups = len(kernel_size)
        self.split_in_channels = split_channels(in_channels, self.num_groups)
        self.split_out_channels = split_channels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)
        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)
        return x


class MDConvBlock(BaseBlock):
    def __init__(self, channels, kernel_size, stride):
        super().__init__(MixNetBlockName.MDConvBlock)

        self.num_groups = len(kernel_size)
        self.split_channels = split_channels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)
        return x


class MixNetBlock(BaseBlock):

    def __init__(self, in_channels, out_channels,
                 kernel_size=(3,), expand_ksize=(1,), project_ksize=(1,),
                 stride=1, expand_ratio=1, se_reduction=0,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(MixNetBlockName.MixNetBlock)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_reduction != 0)
        conv = nn.ModuleList()
        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                NormalizeLayer(bn_name, expand_channels),
                ActivationLayer(activation_name)
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            MDConvBlock(expand_channels, kernel_size, stride),
            NormalizeLayer(bn_name, expand_channels),
            ActivationLayer(activation_name)
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SEConvBlock(expand_channels, in_channels, se_reduction)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            NormalizeLayer(bn_name, out_channels),
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)
