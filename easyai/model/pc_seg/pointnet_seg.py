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
from easyai.model.base_block.pc_cls.pointnet_block import PointNetRouteBlock
from easyai.model.base_block.pc_cls.pointnet_block import PointNetBlockName
from easyai.model.backbone.pc_cls.pointnet import PointNet


class PointNetSeg(BaseModel):

    def __init__(self, data_channel=3, class_num=13):
        super().__init__()
        self.set_name(ModelName.PointNetSeg)

        self.feature_transform = True
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize1d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = PointNet(data_channel=self.data_channel,
                            feature_transform=self.feature_transform,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        route_block = PointNetRouteBlock(1)
        self.add_block_list(route_block.get_name(), route_block, 1024 + 64)

        input_channel = 1024 + 64
        conv1 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=512,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 512)

        input_channel = 512
        conv2 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=256,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 256)

        input_channel = 256
        conv3 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=128,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 128)

        input_channel = 128
        conv4 = nn.Conv1d(input_channel, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional1d, conv4, self.class_number)

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = PointNetSegLoss(self.feature_transform)
        self.add_block_list(loss.get_name(), loss, self.base_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        trans = []
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
                trans = block.get_transform_output()
            elif PointNetBlockName.PointNetRouteBlock in key:
                x = block(x, base_outputs)
            elif LossType.PointNetLoss in key:
                output.append(x)
            else:
                x = block(x)
            print(key, x.shape)
            layer_outputs.append(x)
        output.extend(trans)
        return output
