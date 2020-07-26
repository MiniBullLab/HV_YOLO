#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.darknet_block import ResBlock


__all__ = ['csp_darknet53']


class CSPDarkNet(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(1, 2, 8, 8, 4),
                 out_channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2),
                 dilations=(1, 1, 1, 1, 1), bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.Mish):
        super().__init__(data_channel)
        self.set_name(BackboneName.CSPDarknet53)
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 32
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_darknet_stage(self.out_channels[index], self.num_blocks[index],
                                    self.strides[index], self.dilations[index],
                                    self.bnName, self.activationName)
            self.in_channel = self.block_out_channels[-1]

    def make_darknet_stage(self, out_channels, num_blocks,
                           stride, dilation, bnName, activationName, first=False):
        downsample_conv = ConvBNActivationBlock(in_channels=self.in_channel,
                                                out_channels=out_channels,
                                                kernel_size=3,
                                                stride=stride,
                                                padding=dilation,
                                                dilation=dilation,
                                                bnName=bnName,
                                                activationName=activationName)
        name = "down_%s" % downsample_conv.get_name()
        self.add_block_list(name, downsample_conv, out_channels)
        if first:
            split_conv0 = ConvBNActivationBlock(out_channels, out_channels, 1,
                                                bnName=bnName,
                                                activationName=activationName)
            self.add_block_list(split_conv0.get_name(), split_conv0, out_channels)
            split_conv1 = ConvBNActivationBlock(out_channels, out_channels, 1,
                                                bnName=bnName,
                                                activationName=activationName)
            self.add_block_list(split_conv1.get_name(), split_conv1, out_channels)
            blocks_conv = nn.Sequential(
                ResBlock(channels=out_channels, hidden_channels=out_channels // 2,
                         bnName=bnName,
                         activationName=activationName),
                ConvBNActivationBlock(out_channels, out_channels, 1,
                                      bnName=bnName,
                                      activationName=activationName)
            )
            name = "CSPBlock"
            self.add_block_list(name, blocks_conv, out_channels)
            route = RouteLayer("-1,-3")
            self.add_block_list(route.get_name(), route, out_channels * 2)
            concat_conv = ConvBNActivationBlock(out_channels * 2, out_channels, 1,
                                                bnName=bnName,
                                                activationName=activationName)
            self.add_block_list(concat_conv.get_name(), concat_conv, out_channels)
        else:
            split_conv0 = ConvBNActivationBlock(out_channels, out_channels // 2, 1,
                                                bnName=bnName,
                                                activationName=activationName)
            self.add_block_list(split_conv0.get_name(), split_conv0, out_channels // 2)
            split_conv1 = ConvBNActivationBlock(out_channels, out_channels // 2, 1,
                                                bnName=bnName,
                                                activationName=activationName)
            self.add_block_list(split_conv1.get_name(), split_conv1, out_channels // 2)
            blocks_conv = nn.Sequential(
                *[ResBlock(out_channels // 2, bnName=bnName,
                           activationName=activationName) for _ in range(num_blocks)],
                ConvBNActivationBlock(out_channels // 2, out_channels // 2, 1)
            )
            name = "CSPBlock"
            self.add_block_list(name, blocks_conv, out_channels // 2)
            route = RouteLayer("-1,-3")
            self.add_block_list(route.get_name(), route, out_channels)
            concat_conv = ConvBNActivationBlock(out_channels, out_channels, 1,
                                                bnName=bnName,
                                                activationName=activationName)
            self.add_block_list(concat_conv.get_name(), concat_conv, out_channels)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def csp_darknet53(data_channel):
    model = CSPDarkNet(data_channel=data_channel,
                       num_blocks=[1, 2, 8, 8, 4])
    model.set_name(BackboneName.CSPDarknet53)
    return model
