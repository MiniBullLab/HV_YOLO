#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import LayerType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.activation_function import ActivationFunction
from easyai.model.base_block.utility.normalization_layer import NormalizationFunction


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
        assert len(self.layers) >= 2

    def forward(self, layer_outputs, base_outputs):
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = temp_layer_outputs[0]
        for layer in temp_layer_outputs[1:]:
             x = x * layer
        return x


class AddLayer(BaseBlock):

    def __init__(self, layers):
        super().__init__(LayerType.AddLayer)
        self.layers = [int(x) for x in layers.split(',') if x]
        assert len(self.layers) >= 2

    def forward(self, layer_outputs, base_outputs):
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = temp_layer_outputs[0]
        for layer in temp_layer_outputs[1:]:
             x = x + layer
        return x


class NormalizeLayer(BaseBlock):

    def __init__(self, bn_name, out_channel):
        super().__init__(LayerType.NormalizeLayer)
        self.normalize = NormalizationFunction.get_function(bn_name, out_channel)

    def forward(self, x):
        x = self.normalize(x)
        return x


class ActivationLayer(BaseBlock):

    def __init__(self, activation_name, inplace=True):
        super().__init__(LayerType.ActivationLayer)
        self.activation = ActivationFunction.get_function(activation_name, inplace)

    def forward(self, x):
        x = self.activation(x)
        return x


class RouteLayer(BaseBlock):

    def __init__(self, layers):
        super().__init__(LayerType.RouteLayer)
        self.layers = [int(x) for x in layers.split(',') if x]

    def forward(self, layer_outputs, base_outputs):
        # print(self.layers)
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


class FcLayer(BaseBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(LayerType.FcLayer)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MeanLayer(BaseBlock):
    def __init__(self, dim, keep_dim=False):
        super().__init__(LayerType.MeanLayer)
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(self.dim, self.keep_dim)


if __name__ == "__main__":
    pass
