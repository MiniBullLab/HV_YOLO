#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *


class Detection2dBlock(BaseBlock):

    def __init__(self, input_channle, anchor_number, class_number):
        super().__init__(BlockType.Detection2dBlock)
        self.anchor_number = anchor_number
        self.class_number = class_number
        self.loc_layer = nn.Conv2d(input_channle, anchor_number * 4,
                                   kernel_size=3, padding=1, stride=1)
        self.cls_layer = nn.Conv2d(input_channle, anchor_number * class_number,
                                   kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        loc_cls_result = []
        y_loc = self.loc_layer(x)
        y_cls = self.cls_layer(x)
        loc_cls_result.append(y_loc)
        loc_cls_result.append(y_cls)
        return loc_cls_result
