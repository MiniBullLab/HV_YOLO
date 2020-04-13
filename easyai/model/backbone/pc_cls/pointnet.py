#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock1d
from easyai.model.base_block.utility.utility_block import ConvBNBlock1d
from easyai.model.base_block.pc_cls.pointnet_block import PointNetBlockName
from easyai.model.base_block.pc_cls.pointnet_block import TransformBlock, MaxPool1dBlock


__all__ = ['PointNet']


class PointNet(BaseBackbone):
    def __init__(self, data_channel=3, feature_transform=True,
                 bn_name=NormalizationType.BatchNormalize1d,
                 activation_name=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.PointNet)
        self.data_channel = data_channel
        self.feature_transform = feature_transform
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.in_channel = self.data_channel
        self.transform_outputs = []

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        transform1 = TransformBlock(input_channle=self.data_channel,
                                    bn_name=self.bn_name,
                                    activation_name=self.activation_name)
        self.add_block_list(transform1.get_name(), transform1, self.data_channel)
        self.in_channel = self.data_channel

        conv1 = ConvBNActivationBlock1d(in_channels=self.in_channel,
                                        out_channels=64,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 64)
        self.in_channel = 64

        if self.feature_transform:
            transform1 = TransformBlock(input_channle=self.in_channel,
                                        bn_name=self.bn_name,
                                        activation_name=self.activation_name)
            self.add_block_list(transform1.get_name(), transform1, self.in_channel)

        conv2 = ConvBNActivationBlock1d(in_channels=self.in_channel,
                                        out_channels=128,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 128)
        self.in_channel = 128

        conv3 = ConvBNBlock1d(in_channels=self.in_channel,
                              out_channels=1024,
                              kernel_size=1,
                              bnName=self.bn_name)
        self.add_block_list(conv3.get_name(), conv3, 1024)
        self.in_channel = 1024

        maxpool = MaxPool1dBlock(self.in_channel)
        self.add_block_list(maxpool.get_name(), maxpool, self.in_channel)

    def get_transform_output(self):
        return self.transform_outputs

    def forward(self, x):
        output_list = []
        self.transform_outputs = []
        for key, block in self._modules.items():
            if PointNetBlockName.TransformBlock in key:
                x, trans = block(x)
                self.transform_outputs.append(trans)
            else:
                x = block(x)
            # print(key, x.shape)
            output_list.append(x)
        return output_list
