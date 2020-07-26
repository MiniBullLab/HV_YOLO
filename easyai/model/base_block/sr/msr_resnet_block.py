#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvActivationBlock


class MSRResNetBlockName():

    ResidualBlockNoBN = "ResidualBlockNoBN"


class ResidualBlockNoBN(BaseBlock):

    def __init__(self, in_channel=64, activation_name=ActivationType.ReLU):
        super().__init__(MSRResNetBlockName.ResidualBlockNoBN)
        self.conv1 = ConvActivationBlock(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                         bias=True,
                                         activationName=activation_name)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return identity + out
