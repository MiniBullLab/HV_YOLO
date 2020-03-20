#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import ActivationLayer


class RefineNetBlockName():

    CRPBlock = "CRPBlock"
    RefineNetBlock = "refineNetBlock"


class CRPBlock(BaseBlock):

    def __init__(self, in_planes, out_planes, n_stages):
        super().__init__(RefineNetBlockName.CRPBlock)
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    nn.Conv2d(in_planes if (i == 0) else out_planes,
                              out_planes, 1, stride=1, bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class RefineNetBlock(BaseBlock):

    def __init__(self, in_planes, out_planes, stages,
                 activation_name=ActivationType.ReLU):
        super().__init__(RefineNetBlockName.RefineNetBlock)
        self.activate = ActivationLayer(activation_name, inplace=False)

        self.mflow = CRPBlock(in_planes, out_planes, stages)
        self.conv1 = nn.Conv2d(out_planes, 256, 1, bias=False)
        self.up = Upsample(scale_factor=2, mode='bilinear')

        self.conv2 = nn.Conv2d(256, 256, 1, bias=False)

    def forward(self, x1, x2):
        x1 = self.activate(x1)
        x1 = self.mflow(x1)
        x1 = self.conv1(x1)
        x1 = self.up(x1)

        x2 = self.conv2(x2)
        x = x1 + x2
        return x
