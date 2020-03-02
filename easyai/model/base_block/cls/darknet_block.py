#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class DarknetBlockName():

    ReorgBlock = "reorg"
    BasicBlock = "basicBlock"


class ReorgBlock(BaseBlock):

    def __init__(self, stride=2):
        super().__init__(DarknetBlockName.ReorgBlock)
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % self.stride == 0)
        assert (W % self.stride == 0)
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class BasicBlock(BaseBlock):
    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=NormalizationType.BatchNormalize2d, activationName=ActivationType.ReLU):
        super().__init__(DarknetBlockName.BasicBlock)

        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=planes[0],
                                           kernel_size=3,
                                           stride=stride,
                                           padding=dilation,
                                           dilation=dilation,
                                           bnName=bnName,
                                           activationName=activationName)

        self.conv2 = ConvBNActivationBlock(in_channels=planes[0],
                                           out_channels=planes[1],
                                           kernel_size=3,
                                           stride=stride,
                                           padding=dilation,
                                           dilation=dilation,
                                           bnName=bnName,
                                           activationName=activationName)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
