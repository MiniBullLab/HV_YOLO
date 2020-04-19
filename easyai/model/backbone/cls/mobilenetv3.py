#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.attention_block import SEBlock
from easyai.model.base_block.cls.mobilenet_block import InvertedResidual


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


class MobileNetV3(BaseBackbone):

    def __init__(self, cfgs, mode, data_channel=3, width_mult=1.,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.MobileNetV2_1_0)
        assert mode in ['large', 'small']
        self.cfgs = cfgs
        self.mode = mode
        self.data_channel = data_channel
        self.width_mult = width_mult
        self.activation_name = activationName
        self.bn_name = bnName
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        # building first layer
        output_channel = self.make_divisible(16 * self.width_mult, 8)
        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=output_channel,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=ActivationType.HSwish)
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        input_channel = output_channel
        output_channel = self.make_layer(input_channel, InvertedResidual, self.cfgs)

        # building last several layers
        input_channel = self.block_out_channels[-1]
        layer2 = ConvBNActivationBlock(in_channels=input_channel,
                                       out_channels=output_channel,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=ActivationType.HSwish)
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        if self.mode == 'small':
            layer3 = SEBlock(in_channel=output_channel, reduction=4)
            self.add_block_list(layer3.get_name(), layer3, output_channel)

    def make_layer(self, input_channel, block, cfgs):
        # building inverted residual blocks
        hidden_channel = 0
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = self.make_divisible(c * self.width_mult, 8)
            hidden_channel = self.make_divisible(exp_size * self.width_mult, 8)
            if use_hs == 0:
                temp_block = block(input_channel, exp_size, output_channel, k, s, use_se,
                                   bn_name=self.bn_name, activation_name=ActivationType.ReLU)
            else:
                temp_block = block(input_channel, exp_size, output_channel, k, s, use_se,
                                   bn_name=self.bn_name, activation_name=ActivationType.HSwish)
            self.add_block_list(temp_block.get_name(), temp_block, output_channel)
            input_channel = output_channel
        return hidden_channel

    def make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


def mobilenetv3_large(data_channel):
    """
        Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3, 16, 16, 0, 0, 1],
        [3, 64, 24, 0, 0, 2],
        [3, 72, 24, 0, 0, 1],
        [5, 72, 40, 1, 0, 2],
        [5, 120, 40, 1, 0, 1],
        [5, 120, 40, 1, 0, 1],
        [3, 240, 80, 0, 1, 2],
        [3, 200, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1]
    ]
    model = MobileNetV3(cfgs, mode='large', data_channel=data_channel)
    model.set_name(BackboneName.MobileNetv3_large)
    return model


def mobilenetv3_small(data_channel):
    """
        Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3, 16, 16, 1, 0, 2],
        [3, 72, 24, 0, 0, 2],
        [3, 88, 24, 0, 0, 1],
        [5, 96, 40, 1, 1, 2],
        [5, 240, 40, 1, 1, 1],
        [5, 240, 40, 1, 1, 1],
        [5, 120, 48, 1, 1, 1],
        [5, 144, 48, 1, 1, 1],
        [5, 288, 96, 1, 1, 2],
        [5, 576, 96, 1, 1, 1],
        [5, 576, 96, 1, 1, 1],
    ]
    model = MobileNetV3(cfgs, mode='small', data_channel=data_channel)
    model.set_name(BackboneName.MobileNetv3_small)
    return model



