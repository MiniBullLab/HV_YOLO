#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.base_block.cls.squeezenet_block import FireBlock


__all__ = ['SqueezeNet', 'DilatedSqueezeNet']


class SqueezeNet(BaseBackbone):

    def __init__(self, data_channel=3, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super(SqueezeNet, self).__init__()
        self.set_name(BackboneName.SqueezeNet)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 64

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvActivationBlock(in_channels=self.data_channel,
                                     out_channels=self.first_output,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     dilation=1,
                                     activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)

        planes = (16, 64, 64)
        fire1 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire1.get_name(), fire1, output_channle)

        planes = (16, 64, 64)
        fire2 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire2.get_name(), fire2, output_channle)

        layer3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.add_block_list(LayerType.MyMaxPool2d, layer3, output_channle)

        planes = (32, 128, 128)
        fire3 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire3.get_name(), fire3, output_channle)

        planes = (32, 128, 128)
        fire4 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire4.get_name(), fire4, output_channle)

        layer4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.add_block_list(LayerType.MyMaxPool2d, layer4, output_channle)

        planes = (48, 192, 192)
        fire5 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire5.get_name(), fire5, output_channle)

        planes = (48, 192, 192)
        fire6 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire6.get_name(), fire6, output_channle)

        planes = (64, 256, 256)
        fire7 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire7.get_name(), fire7, output_channle)

        planes = (64, 256, 256)
        fire8 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire8.get_name(), fire8, output_channle)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


class DilatedSqueezeNet(BaseBackbone):
    def __init__(self, data_channel=3, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.DilatedSqueezeNet)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = 64

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvActivationBlock(in_channels=self.data_channel,
                                     out_channels=self.first_output,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     dilation=1,
                                     activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)

        planes = (16, 64, 64)
        fire1 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire1.get_name(), fire1, output_channle)

        planes = (16, 64, 64)
        fire2 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire2.get_name(), fire2, output_channle)

        layer3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self.add_block_list(LayerType.MyMaxPool2d, layer3, output_channle)

        planes = (32, 128, 128)
        fire3 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire3.get_name(), fire3, output_channle)

        planes = (32, 128, 128)
        fire4 = FireBlock(self.block_out_channels[-1], planes, activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire4.get_name(), fire4, output_channle)

        layer4 = nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=False)
        self.add_block_list(LayerType.MyMaxPool2d, layer4, output_channle)

        planes = (48, 192, 192)
        fire5 = FireBlock(self.block_out_channels[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire5.get_name(), fire5, output_channle)

        planes = (48, 192, 192)
        fire6 = FireBlock(self.block_out_channels[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire6.get_name(), fire6, output_channle)

        planes = (64, 256, 256)
        fire7 = FireBlock(self.block_out_channels[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire7.get_name(), fire7, output_channle)

        planes = (64, 256, 256)
        fire8 = FireBlock(self.block_out_channels[-1], planes, dilation=2,
                          activationName=self.activationName)
        output_channle = planes[1] + planes[2]
        self.add_block_list(fire8.get_name(), fire8, output_channle)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list
