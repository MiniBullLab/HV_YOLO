#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.utility_layer import FcLayer
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class GhostNetCls(BaseModel):

    def __init__(self, data_channel=3, class_num=100):
        super().__init__()
        self.set_name(ModelName.GhostNetCls)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.factory.get_base_model(BackboneName.GhostNet, self.data_channel)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.add_block_list(LayerType.GlobalAvgPool, avgpool, base_out_channels[-1])

        output_channel = 1280
        layer1 = FcLayer(base_out_channels[-1], output_channel)
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        layer2 = NormalizeLayer(bn_name=NormalizationType.BatchNormalize1d,
                                out_channel=output_channel)
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        layer3 = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(layer3.get_name(), layer3, output_channel)

        layer4 = nn.Dropout(0.2)
        self.add_block_list(LayerType.Dropout, layer4, output_channel)

        layer5 = nn.Linear(output_channel, self.class_number)
        self.add_block_list(LayerType.FcLinear, layer5, self.class_number)

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
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
