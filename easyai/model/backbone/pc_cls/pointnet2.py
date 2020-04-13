#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.pc_cls.pointnet2_block import PointNet2BlockName
from easyai.model.base_block.pc_cls.pointnet2_block import PointNetSetAbstractionMSG
from easyai.model.base_block.pc_cls.pointnet2_block import PointNetSetAbstractionMRG
from easyai.model.base_block.pc_cls.pointnet2_block import PointNet2RouteBlock
from easyai.model.base_block.pc_cls.pointnet2_block import PointNet2Block
from easyai.model.base_block.pc_cls.pointnet_block import MaxPool1dBlock


__all__ = ['PointNet2']


class PointNet2(BaseBackbone):

    def __init__(self, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__()
        self.set_name(BackboneName.PointNet2)
        self.data_channel = data_channel
        self.activation_name = activation_name
        self.bn_name = bn_name
        self.in_channel = self.data_channel
        self.use_MSG = True

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        if self.use_MSG:
            block1 = PointNetSetAbstractionMSG(512, [0.1, 0.2, 0.4], [16, 32, 128], self.in_channel,
                                               [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            out_channel = 64 + 128 + 128
            self.add_block_list(block1.get_name(), block1, out_channel)

            self.in_channel = out_channel + 3
            block2 = PointNetSetAbstractionMSG(128, [0.2, 0.4, 0.8], [32, 64, 128], self.in_channel,
                                               [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
            out_channel = 128 + 256 + 256
            self.add_block_list(block2.get_name(), block2, out_channel)
        else:
            block1 = PointNetSetAbstractionMRG(npoint=512, radius=0.2, nsample=32,
                                               in_channel=self.in_channel, mlp=[64, 64, 128])
            out_channel = 128
            self.add_block_list(block1.get_name(), block1, out_channel)

            self.in_channel = out_channel + 3
            block2 = PointNetSetAbstractionMRG(npoint=128, radius=0.4, nsample=64,
                                               in_channel=self.in_channel, mlp=[128, 128, 256])
            out_channel = 256
            self.add_block_list(block2.get_name(), block2, out_channel)

        self.in_channel = out_channel + 3
        block3 = PointNet2RouteBlock()
        self.add_block_list(block3.get_name(), block3, self.in_channel)

        block4 = PointNet2Block(self.in_channel, [256, 512, 1024],
                                bn_name=self.bn_name,
                                activation_name=self.activation_name)
        self.add_block_list(block4.get_name(), block4, 1024)

        self.in_channel = 1024
        maxpool = MaxPool1dBlock(self.in_channel)
        self.add_block_list(maxpool.get_name(), maxpool, self.in_channel)

    def forward(self, xyz):
        output_list = []
        points = None
        for key, block in self._modules.items():
            if PointNet2BlockName.PointNetSetAbstraction in key:
                xyz, points = block(xyz, points)
            elif PointNet2BlockName.PointNet2RouteBlock in key:
                points = block(xyz, points)
            else:
                points = block(points)
            # print(key, points.shape)
            output_list.append(points)
        return output_list
