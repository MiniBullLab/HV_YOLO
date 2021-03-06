#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from base_block.base_block import *
from base_name.block_name import BlockType, ActivationType
from base_block.activation_function import ActivationFunction


class EmptyLayer(BaseBlock):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super().__init__(BlockType.EmptyLayer)

    def forward(self, x):
        pass


class RouteLayer(BaseBlock):

    def __init__(self, layers):
        super().__init__(BlockType.RouteLayer)
        self.layers = [int(x) for x in layers.split(',') if x]

    def forward(self, layer_outputs, base_outputs):
        #print(self.layers)
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i] for i in self.layers]
        x = torch.cat(temp_layer_outputs, 1)
        return x


class ShortcutLayer(BaseBlock):

    def __init__(self, layer_from, activationName=ActivationType.Linear):
        super().__init__(BlockType.ShortcutLayer)
        self.layer_from = int(layer_from)
        self.activation = ActivationFunction.get_function(activationName)

    def forward(self, layer_outputs):
        x = layer_outputs[-1] + layer_outputs[self.layer_from]
        x = self.activation(x)
        return x


class Upsample(BaseBlock):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='bilinear'):
        super().__init__(BlockType.Upsample)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class MyMaxPool2d(BaseBlock):

    def __init__(self, kernel_size, stride):
        super().__init__(BlockType.MyMaxPool2d)
        layers = OrderedDict()
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
        if kernel_size == 2 and stride == 1:
            layer1 = nn.ZeroPad2d((0, 1, 0, 1))
            layers["pad2d"] = layer1
            layers[BlockType.MyMaxPool2d] = maxpool
        else:
            layers[BlockType.MyMaxPool2d] = maxpool
        self.layer = nn.Sequential(layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class GlobalAvgPool2d(BaseBlock):
    def __init__(self):
        super().__init__(BlockType.GlobalAvgPool)

    def forward(self, x):
        h, w = x.shape[2:]

        if torch.is_tensor(h) or torch.is_tensor(w):
            h = np.asarray(h)
            w = np.asarray(w)
            return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))
        else:
            return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))


class FcLayer(BaseBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(BlockType.FcLayer)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    pass
