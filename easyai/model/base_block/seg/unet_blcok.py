#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class UNetBlockName():

    DoubleConv2d = "doubleConv"
    DownBlock = "downBlock"
    UpBlock = "upBlock"
    RecurrentBlock = "recurrentBlock"
    RRCNNBlock = "RRCNNBlock"
    AttentionBlock = "AttentionBlock"
    AttentionUpBlock = "AttentionUpBlock"


class DoubleConv2d(BaseBlock):

    def __init__(self, in_channels, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.DoubleConv2d)
        self.double_conv = nn.Sequential(
            ConvBNActivationBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                bnName=bn_name,
                activationName=activation_name),
            ConvBNActivationBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                bnName=bn_name,
                activationName=activation_name)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(BaseBlock):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.DownBlock)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels,
                         bn_name, activation_name)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, mode='bilinear',
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.UpBlock)

        self.up = Upsample(scale_factor=2, mode=mode)
        self.conv = ConvBNActivationBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                bnName=bn_name,
                activationName=activation_name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv(x1)
        # # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # # if you have padding issues, see
        # # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class RecurrentBlock(BaseBlock):
    def __init__(self, out_channels, t=2,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.RecurrentBlock)
        self.t = t
        self.conv = ConvBNActivationBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            bnName=bn_name,
            activationName=activation_name)

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(BaseBlock):
    def __init__(self, ch_in, ch_out, t=2,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.RRCNNBlock)
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t, bn_name=bn_name,
                           activation_name=activation_name),
            RecurrentBlock(ch_out, t=t, bn_name=bn_name,
                           activation_name=activation_name)
        )
        self.conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class AttentionBlock(BaseBlock):
    def __init__(self, F_g, F_l, F_int,
                 bn_name=NormalizationType.BatchNormalize2d):
        super().__init__(UNetBlockName.AttentionBlock)
        self.W_g = ConvBNActivationBlock(
            in_channels=F_g,
            out_channels=F_int,
            kernel_size=1,
            padding=0,
            bias=True,
            bnName=bn_name,
            activationName=ActivationType.Linear)

        self.W_x = ConvBNActivationBlock(
            in_channels=F_l,
            out_channels=F_int,
            kernel_size=1,
            padding=0,
            bias=True,
            bnName=bn_name,
            activationName=ActivationType.Linear)

        self.psi = nn.Sequential(
            ConvBNActivationBlock(
                in_channels=F_int,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=True,
                bnName=bn_name,
                activationName=ActivationType.Linear),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUpBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, mode='bilinear',
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.AttentionUpBlock)

        self.up = Upsample(scale_factor=2, mode=mode)
        self.conv = ConvBNActivationBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                bnName=bn_name,
                activationName=activation_name)
        self.attention = AttentionBlock(F_g=out_channels, F_l=out_channels,
                                        F_int=out_channels // 2,
                                        bn_name=bn_name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv(x1)
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return x
