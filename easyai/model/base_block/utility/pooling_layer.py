#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.base_name.block_name import LayerType
from easyai.model.base_block.utility.base_block import *


class MyMaxPool2d(BaseBlock):

    def __init__(self, kernel_size, stride):
        super().__init__(LayerType.MyMaxPool2d)
        layers = OrderedDict()
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                               padding=int((kernel_size - 1) // 2),
                               ceil_mode=False)
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
