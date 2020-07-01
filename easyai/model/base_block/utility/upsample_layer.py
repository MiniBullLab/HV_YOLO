#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import LayerType
from easyai.model.base_block.utility.base_block import *


class Upsample(BaseBlock):

    def __init__(self, scale_factor=1.0, mode='bilinear', align_corners=False):
        super().__init__(LayerType.Upsample)
        self.scale_factor = scale_factor
        self.mode = mode
        self.image_size = (416, 416)
        self.gain = 1/32
        self.is_onnx_export = False
        if mode in ('nearest', 'area'):
            self.align_corners = None
        else:
            self.align_corners = align_corners
        if self.is_onnx_export:  # explicitly state size, avoid scale_factor
            self.layer = nn.Upsample(size=tuple(int(x * self.gain) for x in self.image_size))
        else:
            self.layer = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode,
                                     align_corners=self.align_corners)

    def forward(self, x):
        x = self.layer(x)
        return x
