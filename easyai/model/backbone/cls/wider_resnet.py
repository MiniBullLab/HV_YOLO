#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.wider_resnet_block import IdentityResidualBlock
from functools import partial

__all__ = ['wider_resnet16', 'wider_resnet20', 'wider_resnet38',
           'wider_resnet16_a2', 'wider_resnet20_a2', 'wider_resnet38_a2']


class WiderResNet(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(1, 1, 1, 1, 1, 1),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.wider_resnet16)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = [(128, 128), (256, 256), (512, 512), (512, 1024),
                             (512, 1024, 2048), (1024, 2048, 4096)]
        self.strides = ()
        self.dilations = ()
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.first_output = 64
        self.in_channels = self.first_output

        if len(num_blocks) != 6:
            raise ValueError("Expected a num_blocks with six values")

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        for mod_id, num in enumerate(self.num_blocks):
            # down
            if mod_id <= 4:
                maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                self.add_block_list(LayerType.MyMaxPool2d, maxpool, self.block_out_channels[-1])
            self.make_block(self.out_channels[mod_id], num,
                            stride=1, dilation=1,
                            bn_name=self.bn_name, activation=self.activation_name)

    def make_block(self, out_channels, num_block, stride, dilation,
                   bn_name, activation):
        for index in range(num_block):
            temp_block = IdentityResidualBlock(self.in_channels,
                                               out_channels,
                                               stride=stride,
                                               dilation=dilation,
                                               bn_name=bn_name,
                                               activation_name=activation)
            if index == 0:
                name = "down_%s" % temp_block.get_name()
            else:
                name = temp_block.get_name()
            self.add_block_list(name, temp_block, out_channels[-1])
            self.add_block_list(temp_block.get_name(), temp_block, out_channels[-1])
            self.in_channels = out_channels[-1]

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


class WiderResNetA2(BaseBackbone):

    def __init__(self, data_channel=3, num_blocks=(1, 1, 1, 1, 1, 1),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.wider_resnet16_a2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks
        self.out_channels = [(128, 128), (256, 256), (512, 512),
                             (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        self.strides = (1, 1, 2, 2, 2, 1)
        self.dilations = (1, 1, 1, 1, 1, 1)
        # self.strides = (1, 1, 2, 1, 1, 1)
        # self.dilations = (1, 1, 1, 2, 4, 4)
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.first_output = 64
        self.in_channels = self.first_output

        if len(num_blocks) != 6:
            raise ValueError("Expected a num_blocks with six values")

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        for mod_id, num in enumerate(self.num_blocks):
            if mod_id == 4:
                drop = partial(nn.Dropout, p=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, p=0.5)
            else:
                drop = None
            if mod_id < 2:
                maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                self.add_block_list(LayerType.MyMaxPool2d, maxpool, self.block_out_channels[-1])
            self.make_block(self.out_channels[mod_id], num, stride=self.strides[mod_id],
                            dilation=self.dilations[mod_id], drop=drop,
                            bn_name=self.bn_name, activation=self.activation_name)

    def make_block(self, out_channels, num_block, stride, dilation,
                   drop, bn_name, activation):
        num_stride = [stride] + [1] * (num_block - 1)
        for index, temp_stride in enumerate(num_stride):
            temp_block = IdentityResidualBlock(self.in_channels,
                                               out_channels,
                                               stride=temp_stride,
                                               dilation=dilation,
                                               dropout=drop,
                                               bn_name=bn_name,
                                               activation_name=activation)
            if index == 0:
                name = "down_%s" % temp_block.get_name()
            else:
                name = temp_block.get_name()
            self.add_block_list(name, temp_block, out_channels[-1])
            self.in_channels = out_channels[-1]

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def wider_resnet16(data_channel):
    model = WiderResNet(data_channel=data_channel,
                        num_blocks=[1, 1, 1, 1, 1, 1])
    model.set_name(BackboneName.wider_resnet16)
    return model


def wider_resnet20(data_channel):
    model = WiderResNet(data_channel=data_channel,
                        num_blocks=[1, 1, 1, 3, 1, 1])
    model.set_name(BackboneName.wider_resnet20)
    return model


def wider_resnet38(data_channel):
    model = WiderResNet(data_channel=data_channel,
                        num_blocks=[3, 3, 6, 3, 1, 1])
    model.set_name(BackboneName.wider_resnet38)
    return model


def wider_resnet16_a2(data_channel):
    model = WiderResNetA2(data_channel=data_channel,
                          num_blocks=[1, 1, 1, 1, 1, 1])
    model.set_name(BackboneName.wider_resnet16_a2)
    return model


def wider_resnet20_a2(data_channel):
    model = WiderResNetA2(data_channel=data_channel,
                          num_blocks=[1, 1, 1, 3, 1, 1])
    model.set_name(BackboneName.wider_resnet20_a2)
    return model


def wider_resnet38_a2(data_channel):
    model = WiderResNetA2(data_channel=data_channel,
                          num_blocks=[3, 3, 6, 3, 1, 1])
    model.set_name(BackboneName.wider_resnet38_a2)
    return model
