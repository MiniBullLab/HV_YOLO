#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
title     = {FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation},
author    = {Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu Yizhou},
booktitle = {arXiv preprint arXiv:1903.11816},
year = {2019}
"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.seg.encnet_loss import EncNetLoss
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.seg.encnet_block import EncNetBlockName
from easyai.model.base_block.seg.encnet_block import JPUBlock, EncBlock, FCNHeadBlock
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class EncNetSeg(BaseModel):

    def __init__(self, data_channel=3, class_num=150):
        super().__init__()
        self.set_name(ModelName.EncNetSeg)
        self.data_channel = data_channel
        self.class_number = class_num
        self.is_jpu = True
        self.lateral = False
        self.is_se_loss = True
        self.is_aux = True
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.factory.get_base_model(BackboneName.ResNet50)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        if self.is_jpu:
            jup = JPUBlock(layers='4,8,14,17', in_planes=(512, 1024, 2048), width=512,
                           bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(jup.get_name(), jup, 512 + 512 + 512 + 512)

        self.enc_head(2048, base_out_channels)

        self.create_loss()

        if self.is_aux:
            route = RouteLayer('14')
            output_channel = sum([base_out_channels[i] if i >= 0
                                  else self.block_out_channels[i] for i in route.layers])
            self.add_block_list(route.get_name(), route, output_channel)

            fcn_head = FCNHeadBlock(1024, self.class_number, 16,
                                    bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(fcn_head.get_name(), fcn_head, self.class_number)

    def enc_head(self, in_channels, base_out_channels):
        if self.is_jpu:
            conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=512,
                                          kernel_size=1,
                                          bias=False,
                                          bnName=self.bn_name,
                                          activationName=self.activation_name)
            self.add_block_list(conv1.get_name(), conv1, 512)
        else:
            conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=512,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False,
                                          bnName=self.bn_name,
                                          activationName=self.activation_name)
            self.add_block_list(conv1.get_name(), conv1, 512)

        if self.lateral:
            route1 = RouteLayer('8')
            output_channel = sum([base_out_channels[i] if i >= 0
                                  else self.block_out_channels[i] for i in route1.layers])
            self.add_block_list(route1.get_name(), route1, output_channel)

            connect1 = ConvBNActivationBlock(in_channels=output_channel,
                                             out_channels=512,
                                             kernel_size=1,
                                             bias=False,
                                             bnName=self.bn_name,
                                             activationName=self.activation_name)
            self.add_block_list(connect1.get_name(), connect1, 512)

            route2 = RouteLayer('14')
            output_channel = sum([base_out_channels[i] if i >= 0
                                  else self.block_out_channels[i] for i in route2.layers])
            self.add_block_list(route2.get_name(), route2, output_channel)

            connect2 = ConvBNActivationBlock(in_channels=output_channel,
                                             out_channels=512,
                                             kernel_size=1,
                                             bias=False,
                                             bnName=self.bn_name,
                                             activationName=self.activation_name)
            self.add_block_list(connect2.get_name(), connect2, 512)

            route3 = RouteLayer('-5,-3,-1')
            output_channel = sum([base_out_channels[i] if i >= 0
                                  else self.block_out_channels[i] for i in route2.layers])
            self.add_block_list(route3.get_name(), route3, output_channel)

            fusion = ConvBNActivationBlock(in_channels=output_channel,
                                           out_channels=512,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False,
                                           bnName=self.bn_name,
                                           activationName=self.activation_name)
            self.add_block_list(fusion.get_name(), fusion, 512)

        encmodule = EncBlock(in_channels=512, nclass=self.class_number, se_loss=self.is_se_loss,
                             bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(encmodule.get_name(), encmodule, 512)

        dropout = nn.Dropout2d(0.1, False)
        self.add_block_list(LayerType.Dropout, dropout, self.block_out_channels[-1])

        conv2 = nn.Conv2d(self.block_out_channels[-1], self.class_number, 1)
        self.add_block_list(LayerType.Convolutional, conv2, self.class_number)

        up = Upsample(scale_factor=8, mode='bilinear')
        self.add_block_list(up.get_name(), up, self.class_number)

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = EncNetLoss(self.class_number, se_loss=self.is_se_loss,
                          aux=self.is_aux, ignore_index=250)
        self.add_block_list(LossType.EncNetLoss, loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        se_loss = None
        aux_loss = None
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif EncNetBlockName.JPUBlock in key:
                x = block(layer_outputs, base_outputs)
            elif EncNetBlockName.EncBlock in key:
                x, se_loss = block(x)
            elif LossType.EncNetLoss in key:
                output.append(x)
            elif EncNetBlockName.FCNHeadBlock in key:
                x = block(x)
                aux_loss = x
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        output.append(aux_loss)
        output.append(se_loss)
        return output

