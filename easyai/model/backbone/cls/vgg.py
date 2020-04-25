#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.utility_block import ConvActivationBlock


__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']


class VGG(BaseBackbone):

    CONFIG_DICT = {
        BackboneName.Vgg11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        BackboneName.Vgg13: [64, 64, 'M', 128, 128, 'M',
                              256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        BackboneName.Vgg16: [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        BackboneName.Vgg19: [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                              512, 512, 512, 512, 'M'],
        'vgg13_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
        'vgg16_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512],
        'vgg19_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512],
        'vgg13_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
        'vgg16_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512],
        'vgg19_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512],
    }

    def __init__(self, data_channel=3, vgg_name=BackboneName.Vgg19,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU, is_norm=False):
        super().__init__()
        self.set_name(BackboneName.Vgg19)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.is_norm = is_norm
        self.vgg_cfg = VGG.CONFIG_DICT.get(vgg_name, None)
        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        self.make_layers(self.vgg_cfg)

    def make_layers(self, cfg):
        in_channels = self.data_channel
        for v in cfg:
            if v == 'M':
                temp_layer = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                self.add_block_list(LayerType.MyMaxPool2d, temp_layer, in_channels)
            else:
                if self.is_norm:
                    conv2d = ConvBNActivationBlock(in_channels=in_channels,
                                                   out_channels=v,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   bnName=self.bnName,
                                                   activationName=self.activationName)
                    self.add_block_list(conv2d.get_name(), conv2d, v)
                else:
                    conv2d = ConvActivationBlock(in_channels=in_channels,
                                                 out_channels=v,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 dilation=1,
                                                 activationName=self.activationName)
                    self.add_block_list(conv2d.get_name(), conv2d, v)
                in_channels = v

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def vgg11(data_channel):
    model = VGG(data_channel=data_channel,
                vgg_name=BackboneName.Vgg11)
    model.set_name(BackboneName.Vgg11)
    return model


def vgg13(data_channel):
    model = VGG(data_channel=data_channel,
                vgg_name=BackboneName.Vgg13)
    model.set_name(BackboneName.Vgg13)
    return model


def vgg16(data_channel):
    model = VGG(data_channel=data_channel,
                vgg_name=BackboneName.Vgg16)
    model.set_name(BackboneName.Vgg16)
    return model


def vgg19(data_channel):
    model = VGG(data_channel=data_channel,
                vgg_name=BackboneName.Vgg19)
    model.set_name(BackboneName.Vgg19)
    return model

