#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ActivationConvBNBlock
from easyai.model.base_block.utility.separable_conv_block import SeperableConv2dBlock


class NasNetBlockName():

    SeperableBranch = "seperableBranch"
    Fit = "fit"
    NormalCell = "normalCell"
    ReductionCell = "reductionCell"


class SeperableBranch(BaseBlock):

    def __init__(self, input_channel, output_channel, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        """Adds 2 blocks of [relu-separable conv-batchnorm]."""
        super().__init__(NasNetBlockName.SeperableBranch)
        self.block1 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2dBlock(in_channels=input_channel,
                                 out_channels=output_channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 bias=bias),
            nn.BatchNorm2d(output_channel)
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2dBlock(in_channels=input_channel,
                                 out_channels=output_channel,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=int(kernel_size / 2)),
            nn.BatchNorm2d(output_channel)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Fit(BaseBlock):
    """Make the cell outputs compatible

    Args:
        prev_filters: filter number of tensor prev, needs to be modified
        filters: filter number of normal cell branch output filters
    """

    def __init__(self, prev_filters, filters,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(NasNetBlockName.Fit)
        self.relu = nn.ReLU()

        self.p1 = nn.Sequential(
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        # make sure there is no information loss
        self.p2 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),  # cropping
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        self.bn = nn.BatchNorm2d(filters)

        self.dim_reduce = ActivationConvBNBlock(in_channels=prev_filters,
                                                out_channels=filters,
                                                kernel_size=1,
                                                bnName=bn_name,
                                                activationName=activation_name)
        self.filters = filters

    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x
        elif x.size(2) != prev.size(2): # image size does not match
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)
        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)
        return prev


class NormalCell(BaseBlock):

    def __init__(self, x_in, prev_in, output_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(NasNetBlockName.NormalCell)

        self.dem_reduce = ActivationConvBNBlock(in_channels=x_in,
                                                out_channels=output_channels,
                                                kernel_size=1,
                                                bias=False,
                                                bnName=bn_name,
                                                activationName=activation_name)
        self.block1_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.block1_right = nn.Sequential()

        self.block2_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.block2_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False)

        self.block3_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block3_right = nn.Sequential()

        self.block4_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block4_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.block5_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False)
        self.block5_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False)

        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x

        # return transformed x as new x, and original x as prev
        # only prev tensor needs to be modified
        prev = self.fit((x, prev))

        h = self.dem_reduce(x)

        x1 = self.block1_left(h) + self.block1_right(h)
        x2 = self.block2_left(prev) + self.block2_right(h)
        x3 = self.block3_left(h) + self.block3_right(h)
        x4 = self.block4_left(prev) + self.block4_right(prev)
        x5 = self.block5_left(prev) + self.block5_right(prev)

        return torch.cat([prev, x1, x2, x3, x4, x5], 1), x


class ReductionCell(BaseBlock):

    def __init__(self, x_in, prev_in, output_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(NasNetBlockName.ReductionCell)

        self.dim_reduce = ActivationConvBNBlock(in_channels=x_in,
                                                out_channels=output_channels,
                                                kernel_size=1,
                                                bnName=bn_name,
                                                activationName=activation_name)

        # block1
        self.layer1block1_left = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)
        self.layer1block1_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        # block2
        self.layer1block2_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1block2_right = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)

        # block3
        self.layer1block3_left = nn.AvgPool2d(3, 2, 1)
        self.layer1block3_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        # block5
        self.layer2block1_left = nn.MaxPool2d(3, 2, 1)
        self.layer2block1_right = SeperableBranch(output_channels, output_channels, 3, stride=1, padding=1)

        # block4
        self.layer2block2_left = nn.AvgPool2d(3, 1, 1)
        self.layer2block2_right = nn.Sequential()

        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))

        h = self.dim_reduce(x)

        layer1block1 = self.layer1block1_left(prev) + self.layer1block1_right(h)
        layer1block2 = self.layer1block2_left(h) + self.layer1block2_right(prev)
        layer1block3 = self.layer1block3_left(h) + self.layer1block3_right(prev)
        layer2block1 = self.layer2block1_left(h) + self.layer2block1_right(layer1block1)
        layer2block2 = self.layer2block2_left(layer1block1) + self.layer2block2_right(layer1block2)

        return torch.cat([
            layer1block2,
            # https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py line 739
            layer1block3,
            layer2block1,
            layer2block2
        ], 1), x
