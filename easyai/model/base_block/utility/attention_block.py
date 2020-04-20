#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer


class SEBlock(BaseBlock):

    def __init__(self, in_channel, reduction=16, activate_name=ActivationType.ReLU):
        super().__init__(BlockType.SEBlock)
        # self.avg_pool = GlobalAvgPool2d()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            ActivationLayer(activate_name),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        # torch.clamp(y, 0, 1)
        return x * y


# ???
class SEConvBlock(BaseBlock):

    def __init__(self, in_channel, reduction=16):
        super().__init__(BlockType.SEBlock)
        self.fc1 = nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        # torch.clamp(y, 0, 1)
        return w