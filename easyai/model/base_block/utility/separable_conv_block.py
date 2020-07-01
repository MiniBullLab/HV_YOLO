#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.base_name.block_name import LayerType
from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_layer import NormalizeLayer
from easyai.model.base_block.utility.utility_block import ActivationConvBNBlock
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class ShuffleBlock(BaseBlock):
    def __init__(self, groups=2):
        super().__init__(BlockType.ShuffleBlock)
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


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


class DepthwiseConv2dBlock(BaseBlock):
    """ DepthwiseConv2D + Normalize2 + Activation """

    def __init__(self, in_channel, kernel_size,
                 padding=0, stride=1, dilation=1, bias=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.DepthwiseConv2dBlock)
        conv = nn.Conv2d(in_channel, in_channel, kernel_size,
                         padding=padding, stride=stride, dilation=dilation,
                         groups=in_channel, bias=bias)
        normal = NormalizeLayer(bn_name, in_channel)
        activation = ActivationLayer(activation_name)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional, conv),
            (bn_name, normal),
            (activation_name, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
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


# MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
class MixConv2dBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5, 7), stride=1, padding=0,
                 dilation=1, bias=True, method='equal_params'):
        super().__init__(BlockType.MixConv2dBlock)

        groups = len(kernel_size)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_channels).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_channels] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.block_list = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                         out_channels=ch[g],
                                         kernel_size=kernel_size[g],
                                         stride=stride,
                                         padding=kernel_size[g] // 2,
                                         dilation=dilation,
                                         bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([block(x) for block in self.block_list], 1)
