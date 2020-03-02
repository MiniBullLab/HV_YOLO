#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
'''EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.efficientnet_block import EfficientNetBlock

__all__ = ['EfficientNet']


class EfficientNet(BaseBackbone):
    def __init__(self, data_channel=3, num_blocks=(1, 2, 2, 3, 3, 4, 1),
                 out_channels=(16, 24, 40, 80, 112, 192, 320),
                 strides=(2, 1, 2, 2, 1, 2, 2), expansions=(1, 6, 6, 6, 6, 6, 6),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.EfficientNet)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.expansions = expansions
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 32
        self.in_channel = self.first_output

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
            self.make_layers(num_block, self.out_channels[index],
                             self.strides[index], self.expansions[index])

    def make_layers(self, num_block, out_planes, stride, expansion):
        strides = [stride] + [1] * (num_block - 1)
        for temp_stride in strides:
            temp_block = EfficientNetBlock(self.in_channel, out_planes, expansion, temp_stride,
                                           bn_name=self.bn_name,
                                           activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, out_planes)
            self.in_channel = out_planes

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list
