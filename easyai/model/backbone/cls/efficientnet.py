#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
'''EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.efficientnet_block import MBConvBlock

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
           'efficientnet_b6', 'efficientnet_b7']


class EfficientNet(BaseBackbone):
    def __init__(self, data_channel=3, width_coef=1., depth_coef=1.,
                 scale=1., dropout_ratio=0.2,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.Swish):
        super().__init__()
        self.set_name(BackboneName.Efficientnet_b0)
        self.data_channel = data_channel
        out_channels = (32, 16, 24, 40, 80, 112, 192, 320, 1280)
        repeats = (1, 2, 2, 3, 3, 4, 1)
        self.out_channels = tuple((round(x * width_coef) for x in out_channels))
        self.repeats = tuple((round(x * depth_coef) for x in repeats))
        self.expands = (1, 6, 6, 6, 6, 6, 6)
        self.strides = (1, 2, 2, 2, 1, 2, 1)
        self.kernel_sizes = (3, 3, 5, 3, 5, 5, 3)
        self.scale = scale
        self.dropout_ratio = dropout_ratio
        self.activation_name = activation_name
        self.bn_name = bn_name

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        if (self.scale < 0.99) or (self.scale > 1.001):
            up = Upsample(scale_factor=self.scale, mode='nearest')
            self.add_block_list(up.get_name(), up, self.data_channel)

        stage1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.out_channels[0],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(stage1.get_name(), stage1, self.out_channels[0])

        for index, number in enumerate(self.repeats):
            self.make_layers(number, self.out_channels[index], self.out_channels[index+1],
                             self.kernel_sizes[index], self.strides[index], self.expands[index])
            activate = ActivationLayer(activation_name=self.activation_name)
            self.add_block_list(activate.get_name(), activate, self.block_out_channels[-1])

        stage9 = ConvBNActivationBlock(in_channels=self.out_channels[7],
                                       out_channels=self.out_channels[8],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(stage9.get_name(), stage9, self.out_channels[8])

    def make_layers(self, num_block, in_planes, out_planes, kernel_size, stride, expansion):
        down_layer = MBConvBlock(in_planes, out_planes, kernel_size, stride,
                                 expand_ratio=expansion, use_se=True, bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        name = "down_%s" % down_layer.get_name()
        self.add_block_list(name, down_layer, out_planes)

        for _ in range(1, num_block):
            temp_layer = MBConvBlock(out_planes, out_planes, kernel_size, 1,
                                     expand_ratio=expansion, use_se=True, bn_name=self.bn_name,
                                     activation_name=self.activation_name)
            self.add_block_list(temp_layer.get_name(), temp_layer, out_planes)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def efficientnet_b0(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.0, depth_coef=1.0,
                         scale=1.0, dropout_ratio=0.2)
    model.set_name(BackboneName.Efficientnet_b0)
    return model


def efficientnet_b1(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.0, depth_coef=1.1,
                         scale=240/224.0, dropout_ratio=0.2)
    model.set_name(BackboneName.Efficientnet_b1)
    return model


def efficientnet_b2(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.1, depth_coef=1.2,
                         scale=260/224.0, dropout_ratio=0.3)
    model.set_name(BackboneName.Efficientnet_b2)
    return model


def efficientnet_b3(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.2, depth_coef=1.4,
                         scale=300/224, dropout_ratio=0.3)
    model.set_name(BackboneName.Efficientnet_b3)
    return model


def efficientnet_b4(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.4, depth_coef=1.8,
                         scale=380/224, dropout_ratio=0.4)
    model.set_name(BackboneName.Efficientnet_b4)
    return model


def efficientnet_b5(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.6, depth_coef=2.2,
                         scale=456/224, dropout_ratio=0.4)
    model.set_name(BackboneName.Efficientnet_b5)
    return model


def efficientnet_b6(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=1.8, depth_coef=2.6,
                         scale=528/224, dropout_ratio=0.5)
    model.set_name(BackboneName.Efficientnet_b6)
    return model


def efficientnet_b7(data_channel):
    model = EfficientNet(data_channel=data_channel, width_coef=2.0, depth_coef=3.1,
                         scale=600/224, dropout_ratio=0.5)
    model.set_name(BackboneName.Efficientnet_b7)
    return model
