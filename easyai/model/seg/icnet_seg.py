#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
ICNet for Real-Time Semantic Segmentation on High-Resolution Images
"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.seg.icnet_block import ICNetBlockName
from easyai.model.base_block.seg.icnet_block import InputDownSample, CascadeFeatureFusion
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class ICNet(BaseModel):

    def __init__(self, data_channel=3, class_num=19):
        super().__init__()
        self.set_name(ModelName.ICNet)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        block1 = InputDownSample()
        self.add_block_list(block1.get_name(), block1, self.data_channel)

        conv1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                      out_channels=32,
                                      kernel_size=3,
                                      padding=1,
                                      stride=2,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 32)

        conv2 = ConvBNActivationBlock(in_channels=32,
                                      out_channels=32,
                                      kernel_size=3,
                                      padding=1,
                                      stride=2,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 32)

        conv3 = ConvBNActivationBlock(in_channels=32,
                                      out_channels=64,
                                      kernel_size=3,
                                      padding=1,
                                      stride=2,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 64)

        backbone1 = self.factory.get_base_model(BackboneName.ResNet50, data_channel=self.data_channel)
        base_out_channels1 = backbone1.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone1, base_out_channels1[-1])

        layer1 = RouteLayer('8')
        output_channel = sum([base_out_channels1[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        backbone2 = self.factory.get_base_model(BackboneName.ResNet50, data_channel=self.data_channel)
        base_out_channels2 = backbone2.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone2, base_out_channels2[-1])

        layer2 = RouteLayer('17')
        output_channel = sum([base_out_channels2[i] if i >= 0
                              else self.block_out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        self.create_head()
        self.create_loss()

    def create_head(self):
        cff_24 = CascadeFeatureFusion(2048, 512, 128, 7, bn_name=self.bn_name,
                                      activation_name=self.activation_name)
        self.add_block_list(cff_24.get_name(), cff_24, 128)

        cff_12 = CascadeFeatureFusion(128, 64, 128, 2, bn_name=self.bn_name,
                                      activation_name=self.activation_name)
        self.add_block_list(cff_12.get_name(), cff_12, 128)

        up1 = Upsample(scale_factor=2, mode='bilinear')
        self.add_block_list(up1.get_name(), up1, 128)

        conv_cls = nn.Conv2d(128, self.class_number, 1, bias=False)
        self.add_block_list(LayerType.Convolutional, conv_cls, self.class_number)

        up2 = Upsample(scale_factor=4, mode='bilinear')
        self.add_block_list(up2.get_name(), up2, self.class_number)

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = CrossEntropy2d(ignore_index=250)
        self.add_block_list(LossType.CrossEntropy2d, loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        input_datas = []
        data_index = 0
        index = 3
        for key, block in self._modules.items():
            if ICNetBlockName.InputDownSample in key:
                input_datas = block(x)
            elif BlockType.BaseNet in key:
                base_outputs = block(input_datas[data_index])
                data_index += 1
            else:
                if LayerType.RouteLayer in key:
                    x = block(layer_outputs, base_outputs)
                elif ICNetBlockName.CascadeFeatureFusion in key:
                    x = block(layer_outputs[-1], layer_outputs[index])
                    index -= 1
                elif LossType.CrossEntropy2d in key:
                    output.append(x)
                else:
                    x = block(x)
                layer_outputs.append(x)
                print(key, layer_outputs[-1].shape)
        return output
