#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class PNASNetBlockName():

    SeparableConv = "separableConv"
    CellA = "cellA"
    CellB = "cellB"


class SeparableConv(BaseBlock):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride,
                 bn_name=NormalizationType.BatchNormalize2d):
        super().__init__(PNASNetBlockName.SeparableConv)
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = NormalizeLayer(bn_name, out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(BaseBlock):
    def __init__(self, in_planes, out_planes, stride=1,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PNASNetBlockName.CellA)
        self.stride = stride
        self.sep_conv1 = SeparableConv(in_planes, out_planes, kernel_size=7,
                                       stride=stride, bn_name=bn_name)
        if stride == 2:
            self.conv1 = ConvBNActivationBlock(in_channels=in_planes,
                                               out_channels=out_planes,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0,
                                               bias=False,
                                               bnName=bn_name,
                                               activationName=ActivationType.Linear)
        self.activate = ActivationLayer(activation_name, inplace=False)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y2 = self.conv1(y2)
        return self.activate(y1+y2)


class CellB(BaseBlock):
    def __init__(self, in_planes, out_planes, stride=1,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PNASNetBlockName.CellB)
        self.stride = stride
        # Left branch
        self.sep_conv1 = SeparableConv(in_planes, out_planes, kernel_size=7,
                                       stride=stride, bn_name=bn_name)
        self.sep_conv2 = SeparableConv(in_planes, out_planes, kernel_size=3,
                                       stride=stride, bn_name=bn_name)
        # Right branch
        self.sep_conv3 = SeparableConv(in_planes, out_planes, kernel_size=5,
                                       stride=stride, bn_name=bn_name)
        if stride==2:
            self.conv1 = ConvBNActivationBlock(in_channels=in_planes,
                                               out_channels=out_planes,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0,
                                               bias=False,
                                               bnName=bn_name,
                                               activationName=ActivationType.Linear)

        self.activate1 = ActivationLayer(activation_name, inplace=False)
        self.activate2 = ActivationLayer(activation_name, inplace=False)

        # Reduce channels
        self.conv2 = ConvBNActivationBlock(in_channels=2 * out_planes,
                                           out_channels=out_planes,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y3 = self.conv1(y3)
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = self.activate1(y1+y2)
        b2 = self.activate2(y3+y4)
        y = torch.cat([b1,b2], 1)
        return self.conv2(y)
