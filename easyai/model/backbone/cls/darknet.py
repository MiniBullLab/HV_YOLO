#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.darknet_block import BasicBlock


__all__ = ['darknet21', 'darknet53',
           'darknet21_dilated8', 'darknet21_dilated16',
           'darknet53_dilated8', 'darknet53_dilated16']


class DarkNet(BaseBackbone):
    def __init__(self, data_channel=3, num_blocks=[1, 2, 8, 8, 4],
                 out_channels=[64, 128, 256, 512, 1024], strides=[2, 2, 2, 2, 2],
                 dilations=[1, 1, 1, 1, 1], bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.LeakyReLU):
        super().__init__()
        self.set_name(BackboneName.Darknet53)
        self.data_channel = data_channel
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
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_darknet_layer(self.out_channels[index], self.num_blocks[index],
                                    self.strides[index], self.dilations[index],
                                    self.bnName, self.activationName)
            self.in_channel = self.block_out_channels[-1]

    def make_darknet_layer(self, out_channel, num_block, stride, dilation,
                          bnName, activationName):
        #downsample
        if dilation > 1:
            if stride == 2:
                stride = 1
                dilation = dilation // 2

        down_layers = ConvBNActivationBlock(in_channels=self.in_channel,
                                            out_channels=out_channel,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=dilation,
                                            dilation=dilation,
                                            bnName=bnName,
                                            activationName=activationName)
        name = "down_%s" % down_layers.get_name()
        self.add_block_list(name, down_layers, out_channel)

        planes = [self.in_channel, out_channel]
        for _ in range(0, num_block):
            layer = BasicBlock(out_channel, planes, stride=1, dilation=dilation,
                               bnName=bnName, activationName=activationName)
            self.add_block_list(layer.get_name(), layer, out_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def darknet21(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 1, 2, 2, 1])
    model.set_name(BackboneName.Darknet21)
    return model


def darknet21_dilated8(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 1, 2, 2, 1],
                    dilations=[1, 1, 1, 2, 4])
    model.set_name(BackboneName.Darknet21_Dilated8)
    return model


def darknet21_dilated16(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 1, 2, 2, 1],
                    dilations=[1, 1, 1, 1, 2])
    model.set_name(BackboneName.Darknet21_Dilated16)
    return model


def darknet53(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 2, 8, 8, 4])
    model.set_name(BackboneName.Darknet53)
    return model


def darknet53_dilated8(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 2, 8, 8, 4],
                    dilations=[1, 1, 1, 2, 4])
    model.set_name(BackboneName.Darknet53_Dilated8)
    return model


def darknet53_dilated16(data_channel):
    model = DarkNet(data_channel=data_channel,
                    num_blocks=[1, 2, 8, 8, 4],
                    dilations=[1, 1, 1, 1, 2])
    model.set_name(BackboneName.Darknet53_Dilated16)
    return model
