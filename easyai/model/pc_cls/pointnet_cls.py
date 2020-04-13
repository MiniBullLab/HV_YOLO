#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.pc_cls.pointnet_loss import PointNetClsLoss
from easyai.model.utility.base_model import *
from easyai.model.base_block.utility.utility_block import FcBNActivationBlock
from easyai.model.base_block.utility.utility_layer import NormalizeLayer
from easyai.model.base_block.utility.utility_layer import ActivationLayer
from easyai.model.backbone.pc_cls.pointnet import PointNet


class PointNetCls(BaseModel):

    def __init__(self, data_channel=3, class_num=40):
        super().__init__()
        self.set_name(ModelName.PointNetCls)

        self.feature_transform = False
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

        input_channel = self.block_out_channels[-1]
        fc1 = FcBNActivationBlock(input_channel, 512,
                                  bnName=self.bn_name,
                                  activationName=self.activation_name)
        self.add_block_list(fc1.get_name(), fc1, 512)

        input_channel = 512
        fc2 = nn.Linear(input_channel, 256)
        self.add_block_list(LayerType.FcLinear, fc2, 256)

        input_channel = 256
        dropout = nn.Dropout(p=0.3)
        self.add_block_list(LayerType.Dropout, dropout, input_channel)

        normalize = NormalizeLayer(self.bn_name, input_channel)
        self.add_block_list(normalize.get_name(), normalize, input_channel)

        activate = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(activate.get_name(), activate, input_channel)

        fc3 = nn.Linear(input_channel, self.class_number)
        self.add_block_list(LayerType.FcLinear, fc3, self.class_number)

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = PointNetClsLoss(self.feature_transform)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
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
            elif LossType.PointNetLoss in key:
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        output.extend(trans)
        return output
