#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.activation_function import ActivationFunction
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class ResidualBlock(BaseBlock):

    def __init__(self, flag, in_channels, out_channels, stride=1, dilation=1, expansion=1,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.ResidualBlock)
        self.residual = nn.Sequential()
        if flag == 0:
            self.residual = nn.Sequential(
                ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bnName=bn_name,
                                      activationName=activation_name),
                ConvBNActivationBlock(in_channels=out_channels,
                                      out_channels=out_channels * expansion,
                                      kernel_size=3,
                                      stride=1,
                                      padding=dilation,
                                      dilation=dilation,
                                      bnName=bn_name,
                                      activationName=activation_name)
            )
        elif flag == 1:
            self.residual = nn.Sequential(
                ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      bnName=bn_name,
                                      activationName=activation_name),
                ConvBNActivationBlock(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bnName=bn_name,
                                      activationName=activation_name),
                ConvBNActivationBlock(in_channels=out_channels,
                                      out_channels=out_channels * expansion,
                                      kernel_size=1,
                                      bnName=bn_name,
                                      activationName=activation_name)
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != expansion * out_channels:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                                  out_channels=out_channels * expansion,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)
        self.activation = ActivationFunction.get_function(activation_name)

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        out = residual + shortcut
        out = self.activation(out)
        return out


class InvertedResidual(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, dilation=1,
                 bnName=NormalizationType.BatchNormalize2d, activationName=ActivationType.ReLU6):
        super().__init__(BlockType.InvertedResidual)
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = OrderedDict()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            convBNReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                                out_channels=inter_channels,
                                                kernel_size=1,
                                                bnName=bnName,
                                                activationName=activationName)
            layer_name = "%s_1" % BlockType.ConvBNActivationBlock
            layers[layer_name] = convBNReLU1
        # dw
        convBNReLU2 = ConvBNActivationBlock(in_channels=inter_channels,
                                            out_channels=inter_channels,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=dilation,
                                            dilation=dilation,
                                            groups=inter_channels,
                                            bnName=bnName,
                                            activationName=activationName)
        # pw-linear
        convBNReLU3 = ConvBNActivationBlock(in_channels=inter_channels,
                                            out_channels=out_channels,
                                            kernel_size=1,
                                            bnName=bnName,
                                            activationName=ActivationType.Linear)

        layer_name = "%s_2" % BlockType.ConvBNActivationBlock
        layers[layer_name] = convBNReLU2
        layer_name = "%s_3" % BlockType.ConvBNActivationBlock
        layers[layer_name] = convBNReLU3
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)
