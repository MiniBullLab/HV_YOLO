#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.utility_layer import FcLayer, ActivationLayer
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class VggNetCls(BaseModel):

    def __init__(self, data_channel=3, class_num=100):
        super().__init__()
        self.set_name(ModelName.VggNetCls)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.factory.get_base_model(BackboneName.Vgg19, self.data_channel)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        # avgpool = nn.AdaptiveAvgPool2d((7, 7))

        layer1 = FcLayer(base_out_channels[-1], 4096)
        self.add_block_list(layer1.get_name(), layer1, 4096)

        layer2 = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(layer2.get_name(), layer2, 4096)

        layer3 = nn.Dropout()
        self.add_block_list(LayerType.Dropout, layer3, 4096)

        layer4 = nn.Linear(4096, 4096)
        self.add_block_list(LayerType.FcLinear, layer4, 4096)

        layer5 = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(layer5.get_name(), layer5, 4096)

        layer6 = nn.Dropout()
        self.add_block_list(LayerType.Dropout, layer6, 4096)

        layer7 = nn.Linear(4096, self.class_number)
        self.add_block_list(LayerType.FcLinear, layer7, self.class_number)

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
