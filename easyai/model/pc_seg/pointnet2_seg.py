#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.pc_cls.pointnet_loss import PointNetSegLoss
from easyai.model.utility.base_model import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock1d
from easyai.model.base_block.pc_cls.pointnet2_block import PointNet2BlockName
from easyai.model.base_block.pc_cls.pointnet2_block import PointNetSetAbstractionMRG
from easyai.model.base_block.pc_cls.pointnet2_block import PointNetFeaturePropagation


class PointNet2Seg(BaseModel):

    def __init__(self, data_channel=3, class_num=13):
        super().__init__()
        self.set_name(ModelName.PointNetSeg)

        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize1d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        input_channel = self.data_channel + 3
        block1 = PointNetSetAbstractionMRG(1024, 0.1, 32, input_channel, [32, 32, 64])
        out_channel = 64
        self.add_block_list(block1.get_name(), block1, out_channel)

        input_channel = out_channel + 3
        block2 = PointNetSetAbstractionMRG(256, 0.2, 32, input_channel, [64, 64, 128])
        out_channel = 128
        self.add_block_list(block2.get_name(), block2, out_channel)

        input_channel = out_channel + 3
        block3 = PointNetSetAbstractionMRG(64, 0.4, 32, input_channel, [128, 128, 256])
        out_channel = 256
        self.add_block_list(block3.get_name(), block3, out_channel)

        input_channel = out_channel + 3
        block4 = PointNetSetAbstractionMRG(16, 0.8, 32, input_channel, [256, 256, 512])
        out_channel = 512
        self.add_block_list(block4.get_name(), block4, out_channel)

        input_channel = 512 + 256
        block5 = PointNetFeaturePropagation(input_channel, [256, 256])
        out_channel = 256
        self.add_block_list(block5.get_name(), block5, out_channel)

        input_channel = out_channel + 128
        block6 = PointNetFeaturePropagation(input_channel, [256, 256])
        out_channel = 256
        self.add_block_list(block6.get_name(), block6, out_channel)

        input_channel = out_channel + 64
        block7 = PointNetFeaturePropagation(input_channel, [256, 128])
        out_channel = 128
        self.add_block_list(block7.get_name(), block7, out_channel)

        input_channel = out_channel
        block8 = PointNetFeaturePropagation(input_channel, [128, 128, 128])
        self.add_block_list(block8.get_name(), block8, out_channel)

        conv1 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=128,
                                        kernel_size=1,
                                        bnName=NormalizationType.BatchNormalize1d,
                                        activationName=ActivationType.ReLU)
        out_channel = 128
        self.add_block_list(conv1.get_name(), conv1, out_channel)

        drop = nn.Dropout(0.5)
        self.add_block_list(LayerType.Dropout, drop, out_channel)

        input_channel = out_channel
        conv2 = nn.Conv1d(input_channel, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional1d, conv2, self.class_number)

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = PointNetSegLoss(False)
        self.add_block_list(loss.get_name(), loss, self.base_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, xyz):
        xyz_outputs = []
        layer_outputs = []
        output = []
        points = xyz[:, :, :]
        xyz_index = 3
        for key, block in self._modules.items():
            if PointNet2BlockName.PointNetSetAbstraction in key:
                xyz_output, points = block(xyz, points)
                xyz_outputs.append(xyz_output)
            elif PointNet2BlockName.PointNetFeaturePropagation in key:
                if xyz_index == 0:
                    points = block(xyz, xyz_outputs[xyz_index],
                                   None, layer_outputs[-1])
                else:
                    points = block(xyz_outputs[xyz_index - 1], xyz_outputs[xyz_index],
                                   layer_outputs[xyz_index - 1], layer_outputs[-1])
                layer_outputs.append(points)
                xyz_index -= 1
            elif LossType.PointNetLoss in key:
                output.append(points)
            else:
                points = block(points)
            layer_outputs.append(points)
            print(key, points.shape)
        return output
