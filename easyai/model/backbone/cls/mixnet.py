#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.mixnet_block import MixNetBlock

__all__ = ['mixnet_s', 'mixnet_m', 'mixnet_l']


class MixNet(BaseBackbone):

    def __init__(self, cfgs, net_type='mixnet_s', data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d):
        super(MixNet, self).__init__()
        self.set_name(BackboneName.mixnet_l)
        self.depth_multiplier = 1.0
        if net_type == 'mixnet_s':
            self.stem_channels = 16
            dropout_rate = 0.2
        elif net_type == 'mixnet_m':
            self.stem_channels = 24
            dropout_rate = 0.25
        elif net_type == 'mixnet_l':
            self.stem_channels = 24
            self.depth_multiplier *= 1.3
            dropout_rate = 0.25
        else:
            raise TypeError('Unsupported MixNet type')

        self.cfgs = cfgs
        self.data_channel = data_channel
        self.feature_size = 1536
        self.bn_name = bn_name
        self.create_block_list()

        # depth multiplier
        if self.depth_multiplier != 1.0:
            self.stem_channels = self.round_channels(self.stem_channels * self.depth_multiplier)
            for i, conf in enumerate(self.cfgs):
                conf_ls = list(conf)
                conf_ls[0] = self.round_channels(conf_ls[0] * self.depth_multiplier)
                conf_ls[1] = self.round_channels(conf_ls[1] * self.depth_multiplier)
                self.cfgs[i] = tuple(conf_ls)

    def create_block_list(self):
        self.clear_list()

        stem_conv = ConvBNActivationBlock(in_channels=self.data_channel,
                                          out_channels=self.stem_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          bias=False,
                                          bnName=self.bn_name,
                                          activationName=ActivationType.ReLU)
        self.add_block_list(stem_conv.get_name(), stem_conv, self.stem_channels)

        self.make_layes(self.cfgs)

        head_conv = ConvBNActivationBlock(in_channels=self.cfgs[-1][1],
                                          out_channels=self.feature_size,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=False,
                                          bnName=self.bn_name,
                                          activationName=ActivationType.ReLU)
        self.add_block_list(head_conv.get_name(), head_conv, self.feature_size)

    def make_layes(self, cfgs):
        # building MixNet blocks
        for in_channels, out_channels, kernel_size, expand_ksize, project_ksize, \
         stride, expand_ratio, non_linear, se_reduction in cfgs:
            temp_block = MixNetBlock(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     expand_ksize=expand_ksize,
                                     project_ksize=project_ksize,
                                     stride=stride,
                                     expand_ratio=expand_ratio,
                                     se_reduction=se_reduction,
                                     bn_name=self.bn_name,
                                     activation_name=non_linear)
            self.add_block_list(temp_block.get_name(), temp_block, out_channels)

    def round_channels(self, c, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
        if new_c < 0.9 * c:
            new_c += divisor
        return new_c

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            print(key, x.shape)
        return output_list


def mixnet_s(data_channel):
    cfgs = [(16, 16, [3], [1], [1], 1, 1, ActivationType.ReLU, 0),
            (16, 24, [3], [1, 1], [1, 1], 2, 6, ActivationType.ReLU, 0),
            (24, 24, [3], [1, 1], [1, 1], 1, 3, ActivationType.ReLU, 0),
            (24, 40, [3, 5, 7], [1], [1], 2, 6, ActivationType.Swish, 2),
            (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40, 80, [3, 5, 7], [1], [1, 1], 2, 6, ActivationType.Swish, 2),
            (80, 80, [3, 5], [1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80, 80, [3, 5], [1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80, 120, [3, 5, 7], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 200, [3, 5, 7, 9, 11], [1], [1], 2, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, ActivationType.Swish, 2)]
    model = MixNet(cfgs, 'mixnet_s', data_channel)
    model.set_name(BackboneName.mixnet_l)
    return model


def mixnet_m(data_channel):
    cfgs = [(24,  24,  [3],          [1],    [1],    1, 1, ActivationType.ReLU,  0),
            (24,  32,  [3, 5, 7],    [1, 1], [1, 1], 2, 6, ActivationType.ReLU,  0),
            (32,  32,  [3],          [1, 1], [1, 1], 1, 3, ActivationType.ReLU,  0),
            (32,  40,  [3, 5, 7, 9], [1],    [1],    2, 6, ActivationType.Swish, 2),
            (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40,  80,  [3, 5, 7],    [1],    [1],    2, 6, ActivationType.Swish, 4),
            (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80,  120, [3],          [1],    [1],    1, 6, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 200, [3, 5, 7, 9], [1],    [1], 2, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, ActivationType.Swish, 2)]
    model = MixNet(cfgs, 'mixnet_m', data_channel)
    model.set_name(BackboneName.mixnet_m)
    return model


def mixnet_l(data_channel):
    cfgs = [(24,  24,  [3],          [1],    [1],    1, 1, ActivationType.ReLU,  0),
            (24,  32,  [3, 5, 7],    [1, 1], [1, 1], 2, 6, ActivationType.ReLU,  0),
            (32,  32,  [3],          [1, 1], [1, 1], 1, 3, ActivationType.ReLU,  0),
            (32,  40,  [3, 5, 7, 9], [1],    [1],    2, 6, ActivationType.Swish, 2),
            (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40,  40,  [3, 5],       [1, 1], [1, 1], 1, 6, ActivationType.Swish, 2),
            (40,  80,  [3, 5, 7],    [1],    [1],    2, 6, ActivationType.Swish, 4),
            (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80,  80,  [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, ActivationType.Swish, 4),
            (80,  120, [3],          [1],    [1],    1, 6, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, ActivationType.Swish, 2),
            (120, 200, [3, 5, 7, 9], [1],    [1], 2, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, ActivationType.Swish, 2),
            (200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, ActivationType.Swish, 2)]
    model = MixNet(cfgs, 'mixnet_l', data_channel)
    model.set_name(BackboneName.mixnet_l)
    return model
