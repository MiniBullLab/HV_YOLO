#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvActivationBlock


class SqueezeNetBlockName():

    FireBlock = "fireBlock"


class FireBlock(BaseBlock):

    def __init__(self, input_channle, planes, stride=1, dilation=1,
                 activationName = ActivationType.ReLU):
        super().__init__(SqueezeNetBlockName.FireBlock)
        self.squeeze = ConvActivationBlock(in_channels=input_channle,
                                           out_channels=planes[0],
                                           kernel_size=1,
                                           stride=stride,
                                           padding=0,
                                           dilation=1,
                                           activationName=activationName)
        self.expand1x1 = ConvActivationBlock(in_channels=planes[0],
                                             out_channels=planes[1],
                                             kernel_size=1,
                                             stride=stride,
                                             padding=0,
                                             dilation=1,
                                             activationName=activationName)
        self.expand3x3 = ConvActivationBlock(in_channels=planes[0],
                                             out_channels=planes[2],
                                             kernel_size=3,
                                             stride=stride,
                                             padding=dilation,
                                             dilation=dilation,
                                             activationName=activationName)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)


