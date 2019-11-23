#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_block.base_block import *
from base_block.activation_function import ActivationFunction
from base_name.block_name import BatchNormType, ActivationType
from base_block.utility_block import ConvBNActivationBlock


class ResnetBlockName():

    BasicBlock = "basicBlock"
    Bottleneck = "bottleneck"


class BasicBlock(BaseBlock):
    expansion = 1

    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__(ResnetBlockName.BasicBlock)

        self.convBnReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=planes,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn2 = ConvBNActivationBlock(in_channels=planes,
                                          out_channels=planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=dilation,
                                          dilation=dilation,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*planes:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.ReLU = ActivationFunction.get_function(activationName)

    def forward(self, x):
        out = self.convBnReLU1(x)
        out = self.convBn2(out)
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out


class Bottleneck(BaseBlock):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__(ResnetBlockName.Bottleneck)

        self.convBnReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=planes,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBnReLU2 = ConvBNActivationBlock(in_channels=planes,
                                          out_channels=planes,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn3 = ConvBNActivationBlock(in_channels=planes,
                                          out_channels=self.expansion*planes,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*planes:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=self.expansion*planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.ReLU = ActivationFunction.get_function(activationName)

    def forward(self, x):
        out = self.convBnReLU1(x)
        out = self.convBnReLU2(out)
        out = self.convBn3(out)
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out
