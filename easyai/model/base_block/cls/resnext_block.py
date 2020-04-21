#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class ResNextBlockName():

    ResNextBottleNeckC = "resNextBottleNeck"


#"""The grouped convolutional layer in Fig. 3(c) performs 32 groups
#of convolutions whose input and output channels are 4-dimensional.
#The grouped convolutional layer concatenates them as the outputs
#of the layer."""
class ResNextBottleNeck(BaseBlock):

    CARDINALITY = 32  # How many groups a feature map was splitted into
    DEPTH = 4
    BASEWIDTH = 64

    def __init__(self, in_channel, out_channel, stride,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(ResNextBlockName.ResNextBottleNeckC)

        C = ResNextBottleNeck.CARDINALITY

        #"""We note that the input/output width of the template is fixed as
        #256-d (Fig. 3), We note that the input/output width of the template
        #is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        #when the feature map is subsampled (see Table 1)."""
        # number of channels per group
        D = int(ResNextBottleNeck.DEPTH * out_channel / ResNextBottleNeck.BASEWIDTH)
        self.split_transforms = nn.Sequential(
            ConvBNActivationBlock(in_channels=in_channel,
                                  out_channels=C * D,
                                  kernel_size=1,
                                  groups=C,
                                  bias=False,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(in_channels=C * D,
                                  out_channels=C * D,
                                  kernel_size=3,
                                  stride=stride,
                                  groups=C,
                                  padding=1,
                                  bias=False,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(in_channels=C * D,
                                  out_channels=out_channel * 4,
                                  kernel_size=1,
                                  bias=False,
                                  bnName=bn_name,
                                  activationName=ActivationType.Linear)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * 4:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channel,
                                                  out_channels=out_channel * 4,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bias=False,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)

        self.relu = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x):
        x1 = self.split_transforms(x)
        x2 = self.shortcut(x)
        x3 = x1 + x2
        x = self.relu(x3)
        return x

