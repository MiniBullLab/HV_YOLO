#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_block.base_block import *
from base_block.utility_layer import GlobalAvgPool2d
from base_block.block_name import BlockType, ActivationType, BatchNormType
from base_block.activation_function import ActivationFunction
from base_block.batchnorm import BatchNormalizeFunction


class ConvActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=True)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (BlockType.Convolutional, conv),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBNActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bnName=BatchNormType.BatchNormalize,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=False)
        bn = BatchNormalizeFunction.get_function(bnName, out_channels)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (BlockType.Convolutional, conv),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class InvertedResidual(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU6):
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


class SEBlock(BaseBlock):
    def __init__(self, channel, reduction=16):
        super().__init__(BlockType.SEBlock)
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == "__main__":
    pass
