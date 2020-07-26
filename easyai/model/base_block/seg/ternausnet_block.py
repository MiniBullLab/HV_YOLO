#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvActivationBlock


class TernausNetBlockName():

    DecoderBlock = "decoderBlock"
    DecoderBlockLinkNet = "decoderBlockLinkNet"


class DecoderBlock(BaseBlock):
    def __init__(self, in_channels, middle_channels, out_channels,
                 activation_name=ActivationType.ReLU, is_deconv=True):
        super().__init__(TernausNetBlockName.DecoderBlock)
        if is_deconv:
            self.block = nn.Sequential(
                ConvActivationBlock(in_channels=in_channels,
                                    out_channels=middle_channels,
                                    kernel_size=3,
                                    padding=1,
                                    activationName=activation_name),
                # nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                #                    padding=1),
                nn.ConvTranspose2d(middle_channels, out_channels,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                ActivationLayer(activation_name),
            )
        else:
            self.block = nn.Sequential(
                Upsample(scale_factor=2, mode='bilinear'),
                ConvActivationBlock(in_channels=in_channels,
                                    out_channels=middle_channels,
                                    kernel_size=3,
                                    padding=1,
                                    activationName=activation_name),
                ConvActivationBlock(in_channels=middle_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=1,
                                    activationName=activation_name),
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlockLinkNet(BaseBlock):
    def __init__(self, in_channels, n_filters):
        super().__init__(TernausNetBlockName.DecoderBlockLinkNet)

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x