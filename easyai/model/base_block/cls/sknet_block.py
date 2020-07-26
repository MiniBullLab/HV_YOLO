#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from functools import reduce
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class SKNetBlockName():

    SKConvBlock = "SKConvBlock"
    SKBlock = "SKBlock"


class SKConv(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=1,
                 M=2, r=16, L=32,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(SKNetBlockName.SKConvBlock)
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for index in range(M):
            self.conv.append(ConvBNActivationBlock(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=3,
                                                   stride=stride,
                                                   padding=1 + index,
                                                   dilation=1 + index,
                                                   groups=32,
                                                   bias=False,
                                                   bnName=bn_name,
                                                   activationName=activation_name))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = ConvBNActivationBlock(in_channels=out_channels,
                                         out_channels=d,
                                         kernel_size=1,
                                         bias=False,
                                         bnName=bn_name,
                                         activationName=activation_name)
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        batch_size = input_data.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i, conv(input_data).size())
            output.append(conv(input_data))
        # the part of fusion
        U = reduce(lambda x, y: x+y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


class SKBlock(BaseBlock):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(SKNetBlockName.SKBlock)
        self.conv1 = ConvBNActivationBlock(in_channels=inplanes,
                                           out_channels=planes,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = SKConv(planes, planes, stride)
        self.conv3 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=planes * self.expansion,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=ActivationType.Linear)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input_data):
        shortcut = input_data
        output = self.conv1(input_data)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.downsample is not None:
            shortcut = self.downsample(input_data)
        output += shortcut
        return self.relu(output)
