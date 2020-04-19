#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer, NormalizeLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.separable_conv_block import SeperableConv2dBlock
from easyai.model.base_block.utility.separable_conv_block import SeparableConv2dBNActivation


class XceptionBlockName():

    EntryFlow = "entryFlow"
    MiddleFLowBlock = "middleFLowBlock"
    ExitFLow = "exitFLow"

    DoubleSeparableConv2dBlock = "doubleSeparableConv2dBlock"
    XceptionConvBlock = "xceptionConvBlock"
    XceptionSumBlock = "xceptionSumBlock"
    XceptionBlock = "xceptionBlock"

    BlockA = "blockA"
    FCAttention = "fcAttention"
    Enc = "enc"


class EntryFlow(BaseBlock):

    def __init__(self, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.EntryFlow)
        self.conv1 = ConvBNActivationBlock(in_channels=data_channel,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=32,
                                           out_channels=64,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv3_residual = nn.Sequential(
            SeperableConv2dBlock(in_channels=64,
                                 out_channels=128,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=128,
                                 out_channels=128,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = ConvBNActivationBlock(in_channels=64,
                                                    out_channels=128,
                                                    kernel_size=1,
                                                    stride=2,
                                                    bias=True,
                                                    bnName=bn_name,
                                                    activationName=ActivationType.Linear)

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=128,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = ConvBNActivationBlock(in_channels=128,
                                                    out_channels=256,
                                                    kernel_size=1,
                                                    stride=2,
                                                    bias=True,
                                                    bnName=bn_name,
                                                    activationName=ActivationType.Linear)

        # no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=256,
                                 out_channels=728,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=728,
                                 out_channels=728,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )

        # no downsampling
        self.conv5_shortcut = ConvBNActivationBlock(in_channels=256,
                                                    out_channels=728,
                                                    kernel_size=1,
                                                    stride=1,
                                                    bias=True,
                                                    bnName=bn_name,
                                                    activationName=ActivationType.Linear)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut
        return x


class MiddleFLowBlock(BaseBlock):

    def __init__(self):
        super().__init__(XceptionBlockName.MiddleFLowBlock)

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=728,
                                 out_channels=728,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=728,
                                 out_channels=728,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=728,
                                 out_channels=728,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual


class ExitFLow(BaseBlock):

    def __init__(self, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.ExitFLow)
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2dBlock(in_channels=728,
                                 out_channels=728,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(728),
            nn.ReLU(),

            SeperableConv2dBlock(in_channels=728,
                                 out_channels=1024,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = ConvBNActivationBlock(in_channels=728,
                                              out_channels=1024,
                                              kernel_size=1,
                                              stride=2,
                                              bias=True,
                                              bnName=bn_name,
                                              activationName=ActivationType.Linear)

        self.conv = nn.Sequential(
            SeperableConv2dBlock(in_channels=1024,
                                 out_channels=1536,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2dBlock(in_channels=1536,
                                 out_channels=2048,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        return output


class DoubleSeparableConv2dBlock(BaseBlock):

    def __init__(self, channel_list, stride=1, dilation=1, relu_first=True,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.DoubleSeparableConv2dBlock)
        self.sep_conv1 = SeparableConv2dBNActivation(inplanes=channel_list[0],
                                                     planes=channel_list[1],
                                                     stride=stride,
                                                     dilation=dilation,
                                                     relu_first=relu_first,
                                                     bn_name=bn_name,
                                                     activation_name=activation_name)
        self.sep_conv2 = SeparableConv2dBNActivation(inplanes=channel_list[1],
                                                     planes=channel_list[2],
                                                     stride=stride,
                                                     dilation=dilation,
                                                     relu_first=relu_first,
                                                     bn_name=bn_name,
                                                     activation_name=activation_name)

    def forward(self, x):
        sc1 = self.sep_conv1(x)
        sc2 = self.sep_conv2(sc1)
        return sc2


class XceptionConvBlock(BaseBlock):

    def __init__(self, channel_list, stride=1, dilation=1, relu_first=True,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.XceptionConvBlock)
        assert len(channel_list) == 4

        self.conv = ConvBNActivationBlock(in_channels=channel_list[0],
                                          out_channels=channel_list[-1],
                                          kernel_size=1,
                                          stride=stride,
                                          bias=False,
                                          bnName=bn_name,
                                          activationName=ActivationType.Linear)
        self.double_sep_conv = DoubleSeparableConv2dBlock(channel_list=channel_list[:3],
                                                          dilation=dilation,
                                                          relu_first=relu_first,
                                                          bn_name=bn_name,
                                                          activation_name=activation_name)
        self.sep_conv = SeparableConv2dBNActivation(inplanes=channel_list[2],
                                                    planes=channel_list[3],
                                                    dilation=dilation,
                                                    stride=stride,
                                                    relu_first=relu_first,
                                                    bn_name=bn_name,
                                                    activation_name=activation_name)
        self.last_inp_channels = channel_list[3]

    def forward(self, inputs):
        sc = self.double_sep_conv(inputs)
        residual = self.sep_conv(sc)
        shortcut = self.conv(inputs)
        outputs = residual + shortcut
        return outputs


class XceptionSumBlock(BaseBlock):
    def __init__(self, channel_list, stride=1, dilation=1, relu_first=True,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.XceptionSumBlock)
        assert len(channel_list) == 4

        self.double_sep_conv = DoubleSeparableConv2dBlock(channel_list=channel_list[:3],
                                                          dilation=dilation,
                                                          relu_first=relu_first,
                                                          bn_name=bn_name,
                                                          activation_name=activation_name)
        self.sep_conv = SeparableConv2dBNActivation(inplanes=channel_list[2],
                                                    planes=channel_list[3],
                                                    dilation=dilation,
                                                    stride=stride,
                                                    relu_first=relu_first,
                                                    bn_name=bn_name,
                                                    activation_name=activation_name)
        self.last_inp_channels = channel_list[3]

    def forward(self, inputs):
        sc = self.double_sep_conv(inputs)
        residual = self.sep_conv(sc)
        outputs = residual + inputs
        return outputs


# -------------------------------------------------
#                   For DFANet
# -------------------------------------------------
class BlockA(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 start_with_relu=True, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.BlockA)
        if out_channels != in_channels or stride != 1:
            self.skip = ConvBNActivationBlock(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=stride,
                                              bias=False,
                                              bnName=bn_name,
                                              activationName=ActivationType.Linear)
        else:
            self.skip = None

        self.relu = ActivationLayer(activation_name, inplace=False)
        rep = list()
        inter_channels = out_channels // 4

        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2dBNActivation(in_channels, inter_channels, 3, 1,
                                               dilation, bn_name=bn_name,
                                               activation_name=activation_name))
        rep.append(NormalizeLayer(bn_name, inter_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2dBNActivation(inter_channels, inter_channels, 3, 1,
                                               dilation, bn_name=bn_name,
                                               activation_name=activation_name))
        rep.append(NormalizeLayer(bn_name, inter_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2dBNActivation(inter_channels, out_channels, 3, stride,
                                                   bn_name=bn_name,
                                                   activation_name=activation_name))
            rep.append(NormalizeLayer(bn_name, out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2dBNActivation(inter_channels, out_channels, 3, 1,
                                                   bn_name=bn_name,
                                                   activation_name=activation_name))
            rep.append(NormalizeLayer(bn_name, out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
        out = out + skip
        return out


class FCAttention(BaseBlock):
    def __init__(self, in_channels, bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(XceptionBlockName.FCAttention)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = ConvBNActivationBlock(in_channels=1000,
                                          out_channels=in_channels,
                                          kernel_size=1,
                                          bias=False,
                                          bnName=bn_name,
                                          activationName=activation_name)

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)
