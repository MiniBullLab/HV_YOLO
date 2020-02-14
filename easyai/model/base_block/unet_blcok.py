#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BatchNormType, ActivationType
from easyai.model.base_block.base_block import *
from easyai.model.base_block.utility_block import ConvBNActivationBlock


class UNetBlockName():

    DoubleConv2d = "doubleConv"
    DownBlock = "downBlock"
    UpBlock =  "upBlock"


class DoubleConv2d(BaseBlock):

    def __init__(self, in_channels, out_channels,
                 bn_name=BatchNormType.BatchNormalize2d,
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
                 bn_name=BatchNormType.BatchNormalize2d,
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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,
                 bn_name=BatchNormType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(UNetBlockName.UpBlock)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv2d(in_channels, out_channels,
                                 bn_name=bn_name,
                                 activation_name=activation_name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)