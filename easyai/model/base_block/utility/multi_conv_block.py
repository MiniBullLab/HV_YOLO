#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class InceptionBlock(BaseBlock):
    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.InceptionBlock)
        # 1x1 conv branch
        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=planes[0],
                                           kernel_size=1,
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation,
                                           bnName=bnName,
                                           activationName=activationName)

        # 1x1 conv -> 3x3 conv branch
        conv2 = ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=planes[1],
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      dilation=dilation,
                                      bnName=bnName,
                                      activationName=activationName)
        conv3 = ConvBNActivationBlock(in_channels=planes[1],
                                      out_channels=planes[2],
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      dilation=dilation,
                                      bnName=bnName,
                                      activationName=activationName)
        self.block2 = nn.Sequential(conv2, conv3)

        # 1x1 conv -> 5x5 conv branch
        conv4 = ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=planes[3],
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      dilation=dilation,
                                      bnName=bnName,
                                      activationName=activationName)
        conv5 = ConvBNActivationBlock(in_channels=planes[3],
                                      out_channels=planes[4],
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      dilation=dilation,
                                      bnName=bnName,
                                      activationName=activationName)
        conv6 = ConvBNActivationBlock(in_channels=planes[4],
                                      out_channels=planes[4],
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      dilation=dilation,
                                      bnName=bnName,
                                      activationName=activationName)
        self.block3 = nn.Sequential(conv4, conv5, conv6)

        # 3x3 pool -> 1x1 conv branch
        pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        conv7 = ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=planes[5],
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      dilation=dilation,
                                      bnName=bnName,
                                      activationName=activationName)
        self.block4 = nn.Sequential(pool1, conv7)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.block2(x)
        y3 = self.block3(x)
        y4 = self.block4(x)
        return torch.cat([y1, y2, y3, y4], 1)
