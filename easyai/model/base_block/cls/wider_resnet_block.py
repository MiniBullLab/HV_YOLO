#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class WiderResNetBlockName():

    IdentityResidualBlock = "IdentityResidualBlock"


class IdentityResidualBlock(BaseBlock):
    """Configurable identity-mapping residual block

            Parameters
            ----------
            in_channels : int
                Number of input channels.
            channels : list of int
                Number of channels in the internal feature maps.
                Can either have two or three elements: if three construct
                a residual block with two `3 x 3` convolutions,
                otherwise construct a bottleneck block with `1 x 1`, then
                `3 x 3` then `1 x 1` convolutions.
            stride : int
                Stride of the first `3 x 3` convolution
            dilation : int
                Dilation to apply to the `3 x 3` convolutions.
            groups : int
                Number of convolution groups.
                This is used to create ResNeXt-style blocks and is only compatible with
                bottleneck blocks.
            dropout: callable
                Function to create Dropout Module.
            dist_bn: Boolean
                A variable to enable or disable use of distributed BN
    """
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 dropout=None,
                 dist_bn=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(WiderResNetBlockName.IdentityResidualBlock)
        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")
        self.dist_bn = dist_bn
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.normal = NormalizeLayer(bn_name, in_channels)
        self.activate = ActivationLayer(activation_name, inplace=False)
        if not is_bottleneck:
            layers = [("conv1", ConvBNActivationBlock(in_channels=in_channels,
                                                      out_channels=channels[0],
                                                      kernel_size=3,
                                                      stride=stride,
                                                      padding=dilation,
                                                      dilation=dilation,
                                                      bias=False,
                                                      bnName=bn_name,
                                                      activationName=activation_name)),
                      ("conv2", nn.Conv2d(channels[0], channels[1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=dilation,
                                          dilation=dilation,
                                          bias=False))
                      ]
            if dropout is not None:
                layers = [layers[0], ("dropout", dropout()), layers[1]]
        else:
            layers = [("conv1", ConvBNActivationBlock(in_channels=in_channels,
                                                      out_channels=channels[0],
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0,
                                                      bias=False,
                                                      bnName=bn_name,
                                                      activationName=activation_name)),
                      ("conv2", ConvBNActivationBlock(in_channels=channels[0],
                                                      out_channels=channels[1],
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=dilation,
                                                      dilation=dilation,
                                                      groups=groups,
                                                      bias=False,
                                                      bnName=bn_name,
                                                      activationName=activation_name)),
                      ("conv3", nn.Conv2d(channels[1], channels[2],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=False))
                      ]
            if dropout is not None:
                layers = [layers[0], layers[1], ("dropout", dropout()), layers[2]]

        self.convs = nn.Sequential(OrderedDict(layers))

        self.shortcut = nn.Sequential()
        if need_proj_conv:
            self.shortcut = nn.Conv2d(in_channels, channels[-1],
                                      kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        bn = self.normal(x)
        bn = self.activate(bn)
        shortcut = self.shortcut(x)
        out = self.convs(bn)
        out = out + shortcut
        return out
