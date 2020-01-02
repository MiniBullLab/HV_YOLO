#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyAI.base_block.base_block import *
from easyAI.base_name.block_name import BatchNormType, ActivationType
from easyAI.base_block.utility_block import ConvBNActivationBlock, SEBlock
import numpy as np


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleNetBlockName():

    ShuffleBlock = "shuffleBlock"
    SplitBlock = "splitBlock"
    BasicBlock = "basicBlock"
    SEBasicBlock = "seBasicBlock"
    DownBlock = "downBlock"

class ShuffleBlock(BaseBlock):
    def __init__(self, groups=2):
        super().__init__(ShuffleNetBlockName.ShuffleBlock)
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C/g), H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(BaseBlock):
    def __init__(self, ratio):
        super().__init__(ShuffleNetBlockName.SplitBlock)
        self.ratio = ratio

    def forward(self, x):
        xChannel = x.size(1)
        if torch.is_tensor(xChannel):
            xChannel = np.asarray(xChannel)

        c = int(xChannel * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

class BasicBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU, split_ratio=0.5):
        super().__init__(ShuffleNetBlockName.BasicBlock)
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)

        self.convBnReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn2 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          groups=in_channels,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.convBnReLU3 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.convBnReLU1(x2)
        out = self.convBn2(out)
        out = self.convBnReLU3(out)
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out

class SEBasicBlock(BaseBlock):
    def __init__(self, inplanes, outplanes, c_tag=0.5, BatchNorm=nn.BatchNorm2d, activation=nn.ReLU, SE=False, residual=False, groups=2, dilation=1):
        super().__init__(ShuffleNetBlockName.SEBasicBlock)
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, padding=dilation, bias=False,
                               groups=self.right_part_out, dilation=dilation)
        self.bn2 = BatchNorm(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(self.right_part_out)
        self.activation = activation

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.residual = residual
        self.groups = groups
        self.SE = SE
        if self.SE:
            self.SELayer = SEBlock(self.right_part_out, 2)

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.SE:
            out = self.SELayer(out)
        if self.residual and self.inplanes == self.outplanes:
            out += right

        return channel_shuffle(torch.cat((left, out), 1), self.groups)

class DownBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=2,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__(ShuffleNetBlockName.DownBlock)
        mid_channels = out_channels // 2
        # left
        self.convBn1l = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=in_channels,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.convBnReLU2l = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=mid_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)
        # right
        self.convBnReLU1r = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=mid_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn2r = ConvBNActivationBlock(in_channels=mid_channels,
                                          out_channels=mid_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=mid_channels,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.convBnReLU3r = ConvBNActivationBlock(in_channels=mid_channels,
                                          out_channels=mid_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        outl = self.convBnReLU2l(self.convBn1l(x))
        # right
        outr = self.convBnReLU3r(self.convBn2r(self.convBnReLU1r(x)))
        # concat
        out = torch.cat([outl, outr], 1)
        out = self.shuffle(out)
        return out