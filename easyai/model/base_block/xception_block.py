#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BatchNormType, ActivationType
from easyai.model.base_block.base_block import *
from easyai.model.base_block.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility_block import SeperableConv2dBlock


class XceptionBlockName():

    EntryFlow = "entryFlow"
    MiddleFLowBlock = "middleFLowBlock"
    ExitFLow = "exitFLow"


class EntryFlow(BaseBlock):

    def __init__(self, data_channel=3,
                 bn_name=BatchNormType.BatchNormalize2d,
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

    def __init__(self, bn_name=BatchNormType.BatchNormalize2d,
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
