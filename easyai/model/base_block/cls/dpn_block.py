#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class DPNBlockName():

    Bottleneck = "bottleneck"


class Bottleneck(BaseBlock):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(DPNBlockName.Bottleneck)
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = ConvBNActivationBlock(in_channels=last_planes,
                                           out_channels=in_planes,
                                           kernel_size=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=in_planes,
                                           out_channels=in_planes,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           groups=32,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv3 = ConvBNActivationBlock(in_channels=in_planes,
                                           out_channels=out_planes+dense_depth,
                                           kernel_size=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=ActivationType.Linear)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = ConvBNActivationBlock(in_channels=last_planes,
                                                  out_channels=out_planes+dense_depth,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bias=False,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out
