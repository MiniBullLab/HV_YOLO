#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""nasnet in pytorch
[1] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le

    Learning Transferable Architectures for Scalable Image Recognition
    https://arxiv.org/abs/1707.07012
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.nasnet_block import NasNetBlockName
from easyai.model.base_block.cls.nasnet_block import NormalCell, ReductionCell

__all__ = ['nasnet']


class NasNet(BaseBackbone):

    def __init__(self, data_channel=3,
                 repeat_cell_num=4, reduction_num=2,
                 filters=44):
        super().__init__()
        self.set_name(BackboneName.NasNet)
        self.data_channel = data_channel
        self.filters = filters
        self.repeat_cell_num = repeat_cell_num
        self.reduction_num = reduction_num

        self.activation_name = ActivationType.ReLU
        self.bn_name = NormalizationType.BatchNormalize2d
        self.first_output = 44
        self.prev_filters = self.first_output
        self.x_filters = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        stem = ConvBNActivationBlock(in_channels=self.data_channel,
                                     out_channels=self.first_output,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False,
                                     bnName=self.bn_name,
                                     activationName=ActivationType.Linear)
        self.add_block_list(stem.get_name(), stem, self.first_output)

        for i in range(self.reduction_num):
            self.make_normal(NormalCell, self.repeat_cell_num, self.filters)
            self.filters *= 2
            self.make_reduction(ReductionCell, self.filters)

        self.make_normal(NormalCell, self.repeat_cell_num, self.filters)

        relu = ActivationLayer(activation_name=self.activation_name, inplace=False)
        self.add_block_list(relu.get_name(), relu, self.filters * 6)

    def make_normal(self, block, repeat, output):
        for _ in range(repeat):
            temp_block = block(self.x_filters, self.prev_filters, output)
            self.prev_filters = self.x_filters
            self.x_filters = output * 6  # concatenate 6 branches
            self.add_block_list(temp_block.get_name(), temp_block, self.x_filters)

    def make_reduction(self, block, output):
        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * 4 #stack for 4 branches
        self.add_block_list(reduction.get_name(), reduction, self.x_filters)

    def forward(self, x):
        output_list = []
        prev = None
        for key, block in self._modules.items():
            if NasNetBlockName.NormalCell in key:
                x, prev = block((x, prev))
            elif NasNetBlockName.ReductionCell in key:
                x, prev = block((x, prev))
            else:
                x = block(x)
            output_list.append(x)
        return output_list
        
        
def nasnet(data_channel):
    # stem filters must be 44, it's a pytorch workaround, cant change to other number
    return NasNet(data_channel=data_channel,
                  repeat_cell_num=4,
                  reduction_num=2,
                  filters=44)

