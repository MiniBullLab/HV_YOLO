#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.deeplab_block import ASPPBlock
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class DeepLabV3(BaseModel):

    def __init__(self, data_channel=3, class_num=19):
        super().__init__()
        self.set_name(ModelName.DeepLabV3)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.factory = BackboneFactory()
        self.create_block_list()
        self.loss_flag = -1

    def create_block_list(self):
        self.clear_list()
        self.lossList = []

        backbone = self.factory.get_base_model(BackboneName.ResNet50)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        head = ASPPBlock(2048, bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(head.get_name(), head, 512)

        out_conv = nn.Conv2d(512, self.class_number, kernel_size=1, stride=1, padding=0, bias=True)
        self.add_block_list(LayerType.Convolutional, out_conv, self.class_number)

        up = Upsample(scale_factor=32, mode='bilinear')
        self.add_block_list(up.get_name(), up, self.block_out_channels[-1])

        self.loss_flag = 0
        self.clear_list()

        self.dsn_head(base_out_channels)

        self.loss_flag = 1
        self.clear_list()

    def dsn_head(self, base_out_channels):
        route = RouteLayer('14')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in route.layers])
        self.add_block_list(route.get_name(), route, output_channel)

        conv1 = ConvBNActivationBlock(in_channels=output_channel,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 512)

        dropout = nn.Dropout2d(0.1)
        self.add_block_list(LayerType.Dropout, dropout, self.block_out_channels[-1])

        out_conv = nn.Conv2d(512, self.class_number, kernel_size=1, stride=1, padding=0, bias=True)
        self.add_block_list(LayerType.Convolutional, out_conv, self.class_number)

        up1 = Upsample(scale_factor=16, mode='bilinear')
        self.add_block_list(up1.get_name(), up1, self.block_out_channels[-1])

    def create_loss(self, input_dict=None):
        if self.loss_flag == 0:
            loss = CrossEntropy2d(ignore_index=250)
            self.add_block_list(LossType.CrossEntropy2d, loss, self.block_out_channels[-1])
            self.lossList.append(loss)
        elif self.loss_flag == 1:
            loss = CrossEntropy2d(ignore_index=250)
            self.add_block_list(LossType.CrossEntropy2d, loss, self.block_out_channels[-1])
            self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        return output