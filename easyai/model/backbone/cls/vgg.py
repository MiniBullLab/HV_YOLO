#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.pooling_layer import MyMaxPool2d
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.backbone.utility.registry import REGISTERED_CLS_BACKBONE


__all__ = ['VGG11', 'VGG13', 'VGG16', 'VGG19']


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
        super().__init__(data_channel)
        self.set_name(BackboneName.Vgg19)
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
                max_pooling = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                self.add_block_list(LayerType.MyMaxPool2d, max_pooling, in_channels)
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


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg11)
class VGG11(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg11)
        self.set_name(BackboneName.Vgg11)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg13)
class VGG13(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg13)
        self.set_name(BackboneName.Vgg13)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg16)
class VGG16(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg16)
        self.set_name(BackboneName.Vgg16)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg19)
class VGG19(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg19)
        self.set_name(BackboneName.Vgg19)
