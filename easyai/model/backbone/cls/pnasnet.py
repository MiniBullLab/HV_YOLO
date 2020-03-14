#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
''' PNASNet in PyTorch.
Paper: Progressive Neural Architecture Search
'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.pnasnet_block import CellA, CellB

__all__ = ['pnasnet_A', 'pnasnet_B']


class PNASNet(BaseBackbone):

    def __init__(self, data_channel=3, num_cells=6,
                 num_planes=44, block=CellA,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.PNASNetA)
        self.data_channel = data_channel
        self.num_cells = num_cells
        self.block = block
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = num_planes
        self.in_planes = self.first_output

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

        self.make_layer(self.first_output, self.num_cells)
        self.downsample(self.first_output * 2)

        self.make_layer(self.first_output * 2, self.num_cells)
        self.downsample(self.first_output * 4)

        self.make_layer(self.first_output * 4, self.num_cells)

    def make_layer(self, planes, num_cells):
        for _ in range(num_cells):
            temp_block = self.block(self.in_planes, planes, stride=1,
                                    bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, planes)
            self.in_planes = planes

    def downsample(self, planes):
        down_block = self.block(self.in_planes, planes, stride=2,
                                bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(down_block.get_name(), down_block, planes)
        self.in_planes = planes

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def pnasnet_A(data_channel):
    model = PNASNet(data_channel=data_channel,
                    num_cells=6,
                    num_planes=44,
                    block=CellA)
    model.set_name(BackboneName.PNASNetA)
    return model


def pnasnet_B(data_channel):
    model = PNASNet(data_channel=data_channel,
                    num_cells=6, num_planes=32,
                    block=CellB)
    model.set_name(BackboneName.PNASNetB)
    return model
