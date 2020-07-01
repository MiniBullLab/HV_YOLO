#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.base_name.block_name import LayerType, BlockType
from easyai.model.base_block.utility.base_block import *


class MyMaxPool2d(BaseBlock):

    def __init__(self, kernel_size, stride, ceil_mode=False):
        super().__init__(LayerType.MyMaxPool2d)
        layers = OrderedDict()
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                               padding=int((kernel_size - 1) // 2),
                               ceil_mode=ceil_mode)
        if kernel_size == 2 and stride == 1:
            layer1 = nn.ZeroPad2d((0, 1, 0, 1))
            layers["pad2d"] = layer1
            layers[LayerType.MyMaxPool2d] = maxpool
        else:
            layers[LayerType.MyMaxPool2d] = maxpool
        self.layer = nn.Sequential(layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class GlobalAvgPool2d(BaseBlock):
    def __init__(self):
        super().__init__(LayerType.GlobalAvgPool)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        return x
        # h, w = x.shape[2:]
        # if torch.is_tensor(h) or torch.is_tensor(w):
        #     h = np.asarray(h)
        #     w = np.asarray(w)
        #     return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))
        # else:
        #     return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))


# SPP
class SpatialPyramidPooling(BaseBlock):
    def __init__(self, pool_sizes=(5, 9, 13)):
        super().__init__(BlockType.SpatialPyramidPooling)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2)
                                       for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features
