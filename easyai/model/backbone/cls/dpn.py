#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
'''Dual Path Networks in PyTorch.'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.dpn_block import Bottleneck
from easyai.model.backbone.utility.registry import REGISTERED_CLS_BACKBONE

__all__ = ['DPN26', 'DPN92']


class DPN(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(2, 2, 2, 2),
                 in_planes=(96, 192, 384, 768),
                 out_channels=(256, 512, 1024, 2048),
                 dense_depths=(16, 32, 24, 128),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.DPN26)
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_channels = out_channels
        self.dense_depths = dense_depths
        self.strides = (1, 2, 2, 2)
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 64
        self.last_planes = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_layer(self.in_planes[index], self.out_channels[index], num_block,
                            self.dense_depths[index], self.strides[index])

    def make_layer(self, in_plane, out_plane, num_block, dense_depth, stride):
        strides = [stride] + [1] * (num_block - 1)
        for index, stride in enumerate(strides):
            temp_block = Bottleneck(self.last_planes, in_plane, out_plane,
                                    dense_depth, stride, index == 0)
            self.last_planes = out_plane + (index + 2) * dense_depth
            self.add_block_list(temp_block.get_name(), temp_block, self.last_planes)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.DPN26)
class DPN26(DPN):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         in_planes=(96, 192, 384, 768),
                         out_channels=(256, 512, 1024, 2048),
                         num_blocks=(2, 2, 2, 2),
                         dense_depths=(16, 32, 24, 128))
        self.set_name(BackboneName.DPN26)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.DPN92)
class DPN92(DPN):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         in_planes=(96, 192, 384, 768),
                         out_channels=(256, 512, 1024, 2048),
                         num_blocks=(3, 4, 20, 3),
                         dense_depths=(16, 32, 24, 128))
        self.set_name(BackboneName.DPN92)

