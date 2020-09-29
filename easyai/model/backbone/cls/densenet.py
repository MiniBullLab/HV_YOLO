#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.densenet_block import DenseBlock, TransitionBlock
from easyai.model.backbone.utility.registry import REGISTERED_CLS_BACKBONE


__all__ = ['Densenet121', 'Densenet169', 'Densenet201', 'Densenet161',
           'Densenet121Dilated8', 'Densenet121Dilated16',
           'Densenet169Dilated8', 'Densenet169Dilated16']


class DenseNet(BaseBackbone):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, data_channel=3, growth_rate=32, num_init_features=64,
                 num_blocks=(6, 12, 24, 16), dilations=(1, 1, 1, 1), bn_size=4, drop_rate=0,
                 bnName=NormalizationType.BatchNormalize2d, activationName=ActivationType.ReLU):

        super().__init__(data_channel)
        self.set_name(BackboneName.Densenet121)
        self.num_init_features = num_init_features
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.activationName = activationName
        self.bnName = bnName

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.num_init_features,
                                       kernel_size=7,
                                       stride=2,
                                       padding=3,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.num_init_features)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.add_block_list(LayerType.MyMaxPool2d, layer2, self.num_init_features)

        self.in_channels = self.num_init_features
        for index, num_block in enumerate(self.num_blocks):
            self.make_densenet_layer(num_block, self.dilations[index],
                                     self.bn_size, self.growth_rate,
                                     self.drop_rate, self.bnName, self.activationName)
            self.in_channels = self.block_out_channels[-1]
            if index != len(self.num_blocks) - 1:
                trans = TransitionBlock(in_channel=self.in_channels,
                                        output_channel=self.in_channels // 2,
                                        stride=1,
                                        bnName=self.bnName,
                                        activationName=self.activationName)
                self.add_block_list(trans.get_name(), trans, self.in_channels // 2)
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                self.add_block_list(LayerType.GlobalAvgPool, avg_pool, self.block_out_channels[-1])
                self.in_channels = self.block_out_channels[-1]
        layer3 = NormalizeLayer(bn_name=self.bnName,
                                out_channel=self.in_channels)
        self.add_block_list(layer3.get_name(), layer3, self.in_channels)

        layer4 = ActivationLayer(self.activationName, False)
        self.add_block_list(layer4.get_name(), layer4, self.in_channels)

    def make_densenet_layer(self, num_block, dilation,
                            bn_size, growth_rate, drop_rate,
                            bnName, activation):

        for i in range(num_block):
            temp_input_channel = self.in_channels + i * growth_rate
            layer = DenseBlock(in_channel=temp_input_channel,
                               growth_rate=growth_rate,
                               bn_size=bn_size,
                               drop_rate=drop_rate,
                               stride=1,
                               dilation=dilation,
                               bnName=bnName,
                               activationName=activation)
            temp_output_channel = temp_input_channel + growth_rate
            self.add_block_list(layer.get_name(), layer, temp_output_channel)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet121)
class Densenet121(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 24, 16))
        self.set_name(BackboneName.Densenet121)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet121_Dilated8)
class Densenet121Dilated8(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 24, 16),
                         dilations=(1, 1, 2, 4))
        self.set_name(BackboneName.Densenet121_Dilated8)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet121_Dilated16)
class Densenet121Dilated16(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 24, 16),
                         dilations=(1, 1, 1, 2))
        self.set_name(BackboneName.Densenet121_Dilated16)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet169)
class Densenet169(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 32, 32))
        self.set_name(BackboneName.Densenet169)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet169Dilated8)
class Densenet169Dilated8(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 32, 32),
                         dilations=(1, 1, 2, 4))
        self.set_name(BackboneName.Densenet169Dilated8)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet169Dilated16)
class Densenet169Dilated16(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 32, 32),
                         dilations=(1, 1, 1, 2))
        self.set_name(BackboneName.Densenet169Dilated16)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet161)
class Densenet161(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=96, growth_rate=48,
                         num_blocks=(6, 12, 36, 24))
        self.set_name(BackboneName.Densenet161)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Densenet201)
class Densenet201(DenseNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=64, growth_rate=32,
                         num_blocks=(6, 12, 48, 32))
        self.set_name(BackboneName.Densenet201)




