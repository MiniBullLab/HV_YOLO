#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class PSPNetBlockName():

    PyramidPooling = "pyramidPooling"


# -----------------------------------------------------------------
#                 For PSPNet, fast_scnn
# -----------------------------------------------------------------
class PyramidPooling(BaseBlock):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PSPNetBlockName.PyramidPooling)
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpools.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(ConvBNActivationBlock(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    bnName=bn_name,
                                                    activationName=activation_name)
                              )

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for (avgpool, conv) in zip(self.avgpools, self.convs):
            feats.append(F.interpolate(conv(avgpool(x)), size,
                                       mode='bilinear', align_corners=True))
        return torch.cat(feats, dim=1)
