#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""Pyramid Scene Parsing Network
    Reference:
        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017
"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.seg.pspnet_block import PyramidPooling
from easyai.model.base_block.seg.encnet_block import EncNetBlockName
from easyai.model.base_block.seg.encnet_block import JPUBlock
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class PSPNetSeg(BaseModel):

    def __init__(self, data_channel=3, class_num=2):
        super().__init__()
        self.set_name(ModelName.PSPNetSeg)
        self.data_channel = data_channel
        self.class_number = class_num
        self.is_jpu = True
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.factory.get_base_model(BackboneName.ResNet101)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        if self.is_jpu:
            jup = JPUBlock(layers='4,8,31,34', in_planes=(512, 1024, 2048), width=512,
                           bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(jup.get_name(), jup, 512 + 512 + 512 + 512)
            scale_factor = 8
        else:
            scale_factor = 32

        psp = PyramidPooling(2048, bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(psp.get_name(), psp, 2048 * 2)

        conv1 = ConvBNActivationBlock(in_channels=2048 * 2,
                                      out_channels=512,
                                      kernel_size=3,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 512)

        dropout = nn.Dropout(0.1)
        self.add_block_list(LayerType.Dropout, dropout, self.block_out_channels[-1])

        conv2 = nn.Conv2d(512, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional, conv2, self.class_number)

        layer = Upsample(scale_factor=scale_factor, mode='bilinear')
        self.add_block_list(layer.get_name(), layer, self.block_out_channels[-1])

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
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
            elif EncNetBlockName.JPUBlock in key:
                x = block(layer_outputs, base_outputs)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        return output



