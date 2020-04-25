#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_layer import RouteLayer, AddLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.separable_conv_block import SeparableConv2dBNActivation
from easyai.model.base_block.cls.xception_block import DoubleSeparableConv2dBlock
from easyai.model.base_block.cls.xception_block import XceptionSumBlock, XceptionConvBlock
from easyai.model.base_block.cls.xception_block import BlockA, FCAttention

__all__ = ['Xception65', 'XceptionA']


class Xception65(BaseBackbone):

    def __init__(self, data_channel=3, output_stride=16,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.Xception65)
        self.data_channel = data_channel
        self.output_stride = output_stride
        self.block_number = 16
        self.activation_name = activationName
        self.bn_name = bnName

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        if self.output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
            exit_block_stride = 2
        elif self.output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
            exit_block_stride = 1
        elif self.output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
            exit_block_stride = 1
        else:
            raise NotImplementedError

        self.entry_flow(entry_block3_stride)
        self.middle_flow(middle_block_dilation)
        self.exit_flow(exit_block_stride, exit_block_dilations)

    def entry_flow(self, entry_block3_stride):
        # Entry flow
        conv1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 32)

        conv2 = ConvBNActivationBlock(in_channels=32,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 64)

        block1 = XceptionConvBlock([64, 128, 128, 128], stride=2, bn_name=self.bn_name,
                                   activation_name=self.activation_name)
        self.add_block_list(block1.get_name(), block1, 128)

        double_sep_conv1 = DoubleSeparableConv2dBlock([128, 256, 256], bn_name=self.bn_name,
                                                      activation_name=self.activation_name)
        self.add_block_list(double_sep_conv1.get_name(), double_sep_conv1, 256)

        sep_conv1 = SeparableConv2dBNActivation(inplanes=256,
                                                planes=256,
                                                dilation=1,
                                                stride=2,
                                                bn_name=self.bn_name,
                                                activation_name=self.activation_name)
        self.add_block_list(sep_conv1.get_name(), sep_conv1, 256)

        layer1 = RouteLayer('-3')
        output_channel = sum([self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        conv3 = ConvBNActivationBlock(in_channels=128,
                                      out_channels=256,
                                      kernel_size=1,
                                      stride=2,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=ActivationType.Linear)
        self.add_block_list(conv3.get_name(), conv3, 256)

        layer2 = AddLayer('-3,-1')
        output_channel = 256
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        double_sep_conv2 = DoubleSeparableConv2dBlock([256, 728, 728], bn_name=self.bn_name,
                                                      activation_name=self.activation_name)
        self.add_block_list(double_sep_conv2.get_name(), double_sep_conv2, 728)

        sep_conv2 = SeparableConv2dBNActivation(inplanes=728,
                                                planes=728,
                                                dilation=1,
                                                stride=entry_block3_stride,
                                                bn_name=self.bn_name,
                                                activation_name=self.activation_name)
        self.add_block_list(sep_conv2.get_name(), sep_conv2, 728)

        layer3 = RouteLayer('-3')
        output_channel = sum([self.block_out_channels[i] for i in layer3.layers])
        self.add_block_list(layer3.get_name(), layer3, output_channel)

        conv4 = ConvBNActivationBlock(in_channels=256,
                                      out_channels=728,
                                      kernel_size=1,
                                      stride=entry_block3_stride,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=ActivationType.Linear)
        self.add_block_list(conv4.get_name(), conv4, 728)

        layer4 = AddLayer('-3,-1')
        output_channel = 728
        self.add_block_list(layer4.get_name(), layer4, output_channel)

    def middle_flow(self, middle_block_dilation):
        # Middle flow (16 units)
        for _ in range(self.block_number):
            temp_block = XceptionSumBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, 728)

    def exit_flow(self, exit_block_stride, exit_block_dilations):
        # Exit flow
        block1 = XceptionConvBlock([728, 728, 1024, 1024], stride=exit_block_stride,
                                   dilation=exit_block_dilations[0], bn_name=self.bn_name,
                                   activation_name=self.activation_name)
        self.add_block_list(block1.get_name(), block1, 1024)

        double_sep_conv1 = DoubleSeparableConv2dBlock([1024, 1536, 1536],
                                                      dilation=exit_block_dilations[1],
                                                      stride=1,
                                                      bn_name=self.bn_name,
                                                      activation_name=self.activation_name)
        self.add_block_list(double_sep_conv1.get_name(), double_sep_conv1, 1536)

        sep_conv1 = SeparableConv2dBNActivation(inplanes=1536,
                                                planes=2048,
                                                dilation=exit_block_dilations[1],
                                                stride=1,
                                                bn_name=self.bn_name,
                                                activation_name=self.activation_name)
        self.add_block_list(sep_conv1.get_name(), sep_conv1, 2048)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        for key, block in self._modules.items():
            if LayerType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            else:
                x = block(x)
            layer_outputs.append(x)
            # print(key, x.shape)
        return layer_outputs


class XceptionA(BaseBackbone):
    def __init__(self, data_channel=3,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.XceptionA)
        self.data_channel = data_channel
        self.activation_name = activationName
        self.bn_name = bnName

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        conv1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                      out_channels=8,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 8)

        self.make_layer(8, 48, 4)
        self.make_layer(48, 96, 6)
        self.make_layer(96, 192, 4)

        fca = FCAttention(192, bn_name=self.bn_name,
                          activation_name=self.activation_name)
        self.add_block_list(fca.get_name(), fca, 192)

    def make_layer(self, in_channel, out_channel, num_block):
        down_block = BlockA(in_channel, out_channel, 2,
                            bn_name=self.bn_name, activation_name=self.activation_name)
        name = "down_%s" % down_block.get_name()
        self.add_block_list(name, down_block, out_channel)
        for _ in range(num_block - 1):
            temp_block = BlockA(out_channel, out_channel, 1,
                                bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, out_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list

