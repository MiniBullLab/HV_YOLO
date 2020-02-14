#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.base_name.block_name import LayerType, ActivationType
from easyai.model.base_block.base_block import *
from easyai.model.base_block.activation_function import ActivationFunction
from easyai.model.base_block.batchnorm import BatchNormalizeFunction


class EmptyLayer(BaseBlock):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super().__init__(LayerType.EmptyLayer)

    def forward(self, x):
        pass


class MultiplyLayer(BaseBlock):

    def __init__(self, layers):
        super().__init__(LayerType.MultiplyLayer)
        self.layers = [int(x) for x in layers.split(',') if x]
        if len(self.layers) <= 1:
            print("% input error" % LayerType.MultiplyLayer)

    def forward(self, layer_outputs, base_outputs):
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = temp_layer_outputs[0]
        for layer in temp_layer_outputs:
             x = x * layer
        return x


class AddLayer(BaseBlock):

    def __init__(self, layers):
        super().__init__(LayerType.AddLayer)
        self.layers = [int(x) for x in layers.split(',') if x]
        if len(self.layers) <= 1:
            print("% input error" % LayerType.AddLayer)

    def forward(self, layer_outputs, base_outputs):
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = temp_layer_outputs[0]
        for layer in temp_layer_outputs:
             x = x + layer
        return x


class NormalizeLayer(BaseBlock):

    def __init__(self, bn_name, out_channel):
        super().__init__(LayerType.NormalizeLayer)
        self.normalize = BatchNormalizeFunction.get_function(bn_name, out_channel)

    def forward(self, x):
        x = self.normalize(x)
        return x


class ActivationLayer(BaseBlock):

    def __init__(self, activation_name):
        super().__init__(LayerType.ActivationLayer)
        self.activation = ActivationFunction.get_function(activation_name)

    def forward(self, x):
        x = self.activation(x)
        return x


class RouteLayer(BaseBlock):

    def __init__(self, layers):
        super().__init__(LayerType.RouteLayer)
        self.layers = [int(x) for x in layers.split(',') if x]

    def forward(self, layer_outputs, base_outputs):
        #print(self.layers)
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = torch.cat(temp_layer_outputs, 1)
        return x


class ShortRouteLayer(BaseBlock):

    def __init__(self, layer_from, activationName=ActivationType.Linear):
        super().__init__(LayerType.ShortRouteLayer)
        self.layer_from = int(layer_from)
        self.activation = ActivationFunction.get_function(activationName)

    def forward(self, layer_outputs):
        x = torch.cat([layer_outputs[self.layer_from],
                       layer_outputs[-1]], 1)
        x = self.activation(x)
        return x


class ShortcutLayer(BaseBlock):

    def __init__(self, layer_from, activationName=ActivationType.Linear):
        super().__init__(LayerType.ShortcutLayer)
        self.layer_from = int(layer_from)
        self.activation = ActivationFunction.get_function(activationName)

    def forward(self, layer_outputs):
        x = layer_outputs[-1] + layer_outputs[self.layer_from]
        x = self.activation(x)
        return x


class Upsample(BaseBlock):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='bilinear'):
        super().__init__(LayerType.Upsample)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class MyMaxPool2d(BaseBlock):

    def __init__(self, kernel_size, stride):
        super().__init__(LayerType.MyMaxPool2d)
        layers = OrderedDict()
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
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
        super().__init__(LayerType.FcLayer)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    pass
