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
        if mode in ('nearest', 'area'):
            self.align_corners = None
        else:
            self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)
