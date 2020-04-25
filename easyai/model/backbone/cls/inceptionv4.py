#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

""" inceptionv4 in pytorch


[1] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.cls.inception_block import InceptionStem
from easyai.model.base_block.cls.inception_block import InceptionA, InceptionB, InceptionC
from easyai.model.base_block.cls.inception_block import ReductionA, ReductionB
from easyai.model.base_block.cls.inception_block import InceptionResNetA, InceptionResNetB, InceptionResNetC
from easyai.model.base_block.cls.inception_block import InceptionResNetReductionA, InceptionResNetReductionB


__all__ = ['inceptionv4', 'inception_resnetv2']


class InceptionV4(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(4, 7, 3),
                 out_channels=(192, 224, 256, 384, 96),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):

        super().__init__()
        self.set_name(BackboneName.InceptionV4)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 384

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        stem_block = InceptionStem(self.data_channel, bn_name=self.bn_name,
                                   activation_name=self.activation_name)
        self.add_block_list(stem_block.get_name(), stem_block, self.first_output)

        input_channle = self.first_output
        self.create_inception_module(input_channle, 384, self.num_blocks[0], InceptionA)

        input_channle = 384
        reduction_a = ReductionA(input_channle, self.out_channels,
                                 bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        self.add_block_list(reduction_a.get_name(), reduction_a, reduction_a.output_channel)

        input_channle = reduction_a.output_channel
        self.create_inception_module(input_channle, 1024, self.num_blocks[1], InceptionB)

        input_channle = 1024
        reduction_b = ReductionB(input_channle, bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        self.add_block_list(reduction_b.get_name(), reduction_b, 1536)

        input_channle = 1536
        self.create_inception_module(input_channle, 1536, self.num_blocks[2], InceptionC)

    def create_inception_module(self, input_channel, output_channel,
                                block_num, block):
        for _ in range(block_num):
            inception_block = block(input_channel, bn_name=self.bn_name,
                                    activation_name=self.activation_name)
            self.add_block_list(inception_block.get_name(), inception_block, output_channel)
            input_channel = output_channel

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


class InceptionResNetV2(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(5, 10, 5),
                 out_channels=(256, 256, 384, 384),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.InceptionResNetV2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 384

        self.outChannelList = []
        self.index = 0

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        stem_block = InceptionStem(self.data_channel, bn_name=self.bn_name,
                                   activation_name=self.activation_name)
        self.add_block_list(stem_block.get_name(), stem_block, self.first_output)

        input_channle = self.first_output
        self.create_inception_module(input_channle, 384, self.num_blocks[0], InceptionResNetA)

        input_channle = 384
        reduction_a = InceptionResNetReductionA(input_channle, self.out_channels,
                                                bn_name=self.bn_name,
                                                activation_name=self.activation_name)
        self.add_block_list(reduction_a.get_name(), reduction_a, reduction_a.output_channel)

        input_channle = reduction_a.output_channel
        self.create_inception_module(input_channle, 1154, self.num_blocks[1], InceptionResNetB)

        input_channle = 1154
        reduction_b = InceptionResNetReductionB(input_channle, bn_name=self.bn_name,
                                                activation_name=self.activation_name)
        self.add_block_list(reduction_b.get_name(), reduction_b, 2146)

        input_channle = 2146
        self.create_inception_module(input_channle, 2048, self.num_blocks[2], InceptionResNetC)

    def create_inception_module(self, input_channel, output_channel,
                                block_num, block):
        for _ in range(block_num):
            inception_block = block(input_channel, bn_name=self.bn_name,
                                    activation_name=self.activation_name)
            self.add_block_list(inception_block.get_name(), inception_block, output_channel)
            input_channel = output_channel

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def inceptionv4(data_channel):
    model = InceptionV4(data_channel=data_channel,
                        num_blocks=(4, 7, 3))
    model.set_name(BackboneName.InceptionV4)
    return model


def inception_resnetv2(data_channel):
    model = InceptionResNetV2(data_channel=data_channel,
                              num_blocks=(5, 10, 5))
    model.set_name(BackboneName.InceptionResNetV2)
    return model
