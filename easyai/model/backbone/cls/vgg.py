#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import BatchNormType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility_block import ConvBNActivationBlock


__all__ = ['vgg13', 'vgg16', 'vgg19']


class VGG(BaseBackbone):

    CONFIG_DICT = {
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

    def __init__(self, vgg_name=BackboneName.Vgg19, data_channel=3,
                 bnName=BatchNormType.BatchNormalize2d, activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.Vgg19)
        self.data_channel = data_channel
        self.activationName = activationName
        self.bnName = bnName
        self.vgg_cfg = VGG.CONFIG_DICT.get(vgg_name, None)

        self.outChannelList = []
        self.index = 0

        self.make_layers(self.vgg_cfg)

    def make_layers(self, cfg):
        in_channels = self.data_channel
        for v in cfg:
            if v == 'M':
                temp_layer = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                self.addBlockList(LayerType.MyMaxPool2d, temp_layer, in_channels)
            else:
                conv2d = ConvBNActivationBlock(in_channels=in_channels,
                                               out_channels=v,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               dilation=1,
                                               bnName=self.bnName,
                                               activationName=self.activationName)
                self.addBlockList(conv2d.get_name(), conv2d, v)
                in_channels = v

    def addBlockList(self, blockName, block, out_channel):
        blockName = "base_%s_%d" % (blockName, self.index)
        self.add_module(blockName, block)
        self.outChannelList.append(out_channel)
        self.index += 1

    def getOutChannelList(self):
        return self.outChannelList

    def printBlockName(self):
        for key in self._modules.keys():
            print(key)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def vgg13():
    model = VGG(BackboneName.Vgg13)
    model.set_name(BackboneName.Vgg13)
    return model


def vgg16():
    model = VGG(BackboneName.Vgg16)
    model.set_name(BackboneName.Vgg16)
    return model


def vgg19():
    model = VGG(BackboneName.Vgg19)
    model.set_name(BackboneName.Vgg19)
    return model

