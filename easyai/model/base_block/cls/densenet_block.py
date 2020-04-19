#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import BNActivationConvBlock


class DenseNetBlockName():

    DenseBlock = "denseBlock"
    TransitionBlock = "transitionBlock"


class DenseBlock(BaseBlock):

    def __init__(self, in_channel, growth_rate, bn_size, drop_rate, stride=1, dilation=1,
                 bnName=NormalizationType.BatchNormalize2d, activationName=ActivationType.ReLU):
        super().__init__(DenseNetBlockName.DenseBlock)
        self.layer1 = BNActivationConvBlock(in_channels=in_channel,
                                            out_channels=bn_size*growth_rate,
                                            kernel_size=1,
                                            stride=stride,
                                            padding=0,
                                            dilation=1,
                                            bnName=bnName,
                                            activationName=activationName)
        self.layer2 = BNActivationConvBlock(in_channels=bn_size * growth_rate,
                                            out_channels=growth_rate,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=dilation,
                                            dilation=dilation,
                                            bnName=bnName,
                                            activationName=activationName)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.layer1(x)
        new_features = self.layer2(out)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class TransitionBlock(BaseBlock):

    def __init__(self, in_channel, output_channel, stride=1,
                 bnName=NormalizationType.BatchNormalize2d, activationName=ActivationType.ReLU):
        super().__init__(DenseNetBlockName.TransitionBlock)
        self.block = BNActivationConvBlock(in_channels=in_channel,
                                           out_channels=output_channel,
                                           kernel_size=1,
                                           stride=stride,
                                           padding=0,
                                           dilation=1,
                                           bnName=bnName,
                                           activationName=activationName)

    def forward(self, x):
        x = self.block(x)
        return x

