#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.activation_function import ActivationFunction
from easyai.model.base_block.utility.normalization_layer import NormalizationFunction


class ConvBNActivationBlock1d(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bnName=NormalizationType.BatchNormalize1d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock1d)
        conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional1d, conv),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBNBlock1d(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bnName=NormalizationType.BatchNormalize1d):
        super().__init__(BlockType.ConvBNBlock1d)
        conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional1d, conv),
            (bnName, bn)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class SeperableConv2dBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True):
        super().__init__(BlockType.SeperableConv2dBlock)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                         stride, padding, dilation, in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional, conv),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBNActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional, conv),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        # print(self, x.shape)
        return x


class BNActivationConvBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.BNActivationConvBlock)
        bn = NormalizationFunction.get_function(bnName, in_channels)
        activation = ActivationFunction.get_function(activationName)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        self.block = nn.Sequential(OrderedDict([
            (bnName, bn),
            (activationName, activation),
            (LayerType.Convolutional, conv)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ActivationConvBNBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ActivationConvBNBlock)
        activation = ActivationFunction.get_function(activationName)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        self.block = nn.Sequential(OrderedDict([
            (activationName, activation),
            (LayerType.Convolutional, conv),
            (bnName, bn)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class SeparableConv2dBNActivation(BaseBlock):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.SeparableConv2dBNActivation)

        if relu_first:
            depthwise = ActivationConvBNBlock(in_channels=inplanes,
                                              out_channels=inplanes,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=dilation,
                                              dilation=dilation,
                                              groups=inplanes,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=activation_name)
            pointwise = ConvBNActivationBlock(in_channels=inplanes,
                                              out_channels=planes,
                                              kernel_size=1,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=ActivationType.Linear)
        else:
            depthwise = ConvBNActivationBlock(in_channels=inplanes,
                                              out_channels=inplanes,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=dilation,
                                              dilation=dilation,
                                              groups=inplanes,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=activation_name)
            pointwise = ConvBNActivationBlock(in_channels=inplanes,
                                              out_channels=planes,
                                              kernel_size=1,
                                              bias=bias,
                                              bnName=bn_name,
                                              activationName=activation_name)

        self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                ('pointwise', pointwise)
                                                ]))

    def forward(self, x):
        return self.block(x)


class FcBNActivationBlock(BaseBlock):

    def __init__(self, in_features, out_features, bias=True,
                 bnName=NormalizationType.BatchNormalize1d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.FcBNActivationBlock)
        fc = nn.Linear(in_features, out_features, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_features)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.FcLinear, fc),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


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


class SEBlock(BaseBlock):

    def __init__(self, in_channel, reduction=16):
        super().__init__(BlockType.SEBlock)
        # self.avg_pool = GlobalAvgPool2d()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        # torch.clamp(y, 0, 1)
        return x * y


# ???
class SEConvBlock(BaseBlock):

    def __init__(self, in_channel, reduction=16):
        super().__init__(BlockType.SEBlock)
        self.fc1 = nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        # torch.clamp(y, 0, 1)
        return w


if __name__ == "__main__":
    pass
