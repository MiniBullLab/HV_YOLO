#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.base_model_name import BaseModelName
from easyai.base_name.block_name import BatchNormType, ActivationType, BlockType
from easyai.model.base_block import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility_block import ConvBNActivationBlock
from easyai.model.base_block import DenseBlock, TransitionBlock


__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161',
           'densenet121_dilated8', 'densenet121_dilated16']


class DenseNet(BaseModel):
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
                 bnName=BatchNormType.BatchNormalize2d, activationName=ActivationType.ReLU):

        super().__init__()
        self.set_name(BaseModelName.Densenet121)
        self.data_channel = data_channel
        self.num_init_features = num_init_features
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.activationName = activationName
        self.bnName = bnName

        self.outChannelList = []
        self.index = 0
        self.create_block_list()

    def create_block_list(self):
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.num_init_features,
                                       kernel_size=7,
                                       stride=2,
                                       padding=3,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.addBlockList(layer1.get_name(), layer1, self.num_init_features)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.addBlockList(BlockType.MyMaxPool2d, layer2, self.num_init_features)

        self.in_channels = self.num_init_features
        for index, num_block in enumerate(self.num_blocks):
            self.make_densenet_layer(num_block, self.dilations[index],
                                     self.bn_size, self.growth_rate,
                                     self.drop_rate, self.bnName, self.activationName)
            self.in_channels = self.outChannelList[-1]
            if index != len(self.num_blocks) - 1:
                trans = TransitionBlock(in_channel=self.in_channels,
                                        output_channel=self.in_channels // 2,
                                        stride=1,
                                        bnName=self.bnName,
                                        activationName=self.activationName)
                self.addBlockList(trans.get_name(), trans, self.in_channels // 2)
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                self.addBlockList(BlockType.GlobalAvgPool, avg_pool, self.outChannelList[-1])
                self.in_channels = self.outChannelList[-1]
        layer3 = NormalizeLayer(bn_name=self.bnName,
                                out_channel=self.in_channels)
        self.addBlockList(layer3.get_name(), layer3, self.in_channels)

        layer4 = ActivationLayer(self.activationName)
        self.addBlockList(layer4.get_name(), layer4, self.in_channels)

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
            self.addBlockList(layer.get_name(), layer, temp_output_channel)

    def addBlockList(self, blockName, block, out_channel):
        blockName = "base_%s_%d" % (blockName, self.index)
        self.add_module(blockName, block)
        self.outChannelList.append(out_channel)
        self.index += 1

    def getOutChannelList(self):
        return self.outChannelList

    def printBlockName(self):
        for key in self._modules.keys():
            print(key)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            print(x.shape)
            x = block(x)
            print(key, x.shape)
            output_list.append(x)
        return output_list


def densenet121():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 24, 16))
    model.set_name(BaseModelName.Densenet121)
    return model


def densenet121_dilated8():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 24, 16),
                     dilations=(1, 1, 2, 4))
    model.set_name(BaseModelName.Densenet121_Dilated8)
    return model


def densenet121_dilated16():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 24, 16),
                     dilations=(1, 1, 1, 2))
    model.set_name(BaseModelName.Densenet121_Dilated16)
    return model


def densenet169():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 32, 32))
    model.set_name(BaseModelName.Densenet169)
    return model


def densenet169_dilated8():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 32, 32),
                     dilations=(1, 1, 2, 4))
    model.set_name(BaseModelName.Densenet169)
    return model


def densenet169_dilated16():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 32, 32),
                     dilations=(1, 1, 1, 2))
    model.set_name(BaseModelName.Densenet169)
    return model


def densenet201():
    model = DenseNet(num_init_features=64, growth_rate=32,
                     num_blocks=(6, 12, 48, 32))
    model.set_name(BaseModelName.Densenet201)
    return model


def densenet161():
    model = DenseNet(num_init_features=96, growth_rate=48,
                     num_blocks=(6, 12, 36, 24))
    model.set_name(BaseModelName.Densenet161)
    return model
