#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""xception in pytorch
[1] François Chollet

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.cls.xception_block import EntryFlow, ExitFLow
from easyai.model.base_block.cls.xception_block import MiddleFLowBlock

__all__ = ['xception']


class Xception(BaseBackbone):

    def __init__(self, data_channel=3,
                 block=MiddleFLowBlock,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.Xception)
        self.data_channel = data_channel
        self.block = block
        self.block_number = 8
        self.activation_name = activationName
        self.bn_name = bnName

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        entry_flow = EntryFlow(data_channel=self.data_channel,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(entry_flow.get_name(), entry_flow, 728)

        for _ in range(self.block_number):
            temp_block = self.block()
            self.add_block_list(temp_block.get_name(), temp_block, 728)

        exit_flow = ExitFLow(bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(exit_flow.get_name(), exit_flow, 2048)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def xception(data_channel):
    return Xception(data_channel=data_channel,
                    block=MiddleFLowBlock)


