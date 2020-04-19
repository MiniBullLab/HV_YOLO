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
from easyai.model.base_block.utility.utility_layer import AddLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class FCN8sSeg(BaseModel):

    def __init__(self, class_num=2):
        super().__init__()
        self.set_name(ModelName.FCNSeg)
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        backbone = self.factory.get_base_model(BackboneName.Vgg16)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = ConvBNActivationBlock(in_channels=base_out_channels[-1],
                                       out_channels=4096,
                                       kernel_size=7,
                                       padding=3,
                                       bias=True,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, 4096)

        dropout1 = nn.Dropout()
        self.add_block_list(LayerType.Dropout, dropout1, 4096)

        layer2 = ConvBNActivationBlock(in_channels=4096,
                                       out_channels=4096,
                                       kernel_size=1,
                                       padding=0,
                                       bias=True,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, 4096)

        dropout2 = nn.Dropout()
        self.add_block_list(LayerType.Dropout, dropout2, 4096)

        layer3 = nn.Conv2d(4096, self.class_number, kernel_size=1)
        self.add_block_list(LayerType.Convolutional, layer3, self.class_number)

        upsample1 = Upsample(scale_factor=2, mode='bilinear')
        self.add_block_list(LayerType.Upsample, upsample1, self.block_out_channels[-1])

        layer4 = RouteLayer('13')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer4.layers])
        self.add_block_list(layer4.get_name(), layer4, output_channel)

        layer5 = nn.Conv2d(512, self.class_number, kernel_size=1)
        self.add_block_list(LayerType.Convolutional, layer5, self.class_number)

        layer6 = AddLayer('-1,-3')
        index = layer6.layers[0]
        output_channel = base_out_channels[index] if index >= 0 else self.block_out_channels[index]
        self.add_block_list(layer6.get_name(), layer6, output_channel)

        layer7 = RouteLayer('9')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer7.layers])
        self.add_block_list(layer7.get_name(), layer7, output_channel)

        layer8 = nn.Conv2d(256, self.class_number, kernel_size=1)
        self.add_block_list(LayerType.Convolutional, layer8, self.class_number)

        layer9 = RouteLayer('-3')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer7.layers])
        self.add_block_list(layer9.get_name(), layer9, output_channel)

        upsample2 = Upsample(scale_factor=2, mode='bilinear')
        self.add_block_list(LayerType.Upsample, upsample2, self.block_out_channels[-1])

        layer10 = AddLayer('-1,-3')
        index = layer10.layers[0]
        output_channel = base_out_channels[index] if index >= 0 else self.block_out_channels[index]
        self.add_block_list(layer10.get_name(), layer10, output_channel)

        upsample3 = Upsample(scale_factor=8, mode='bilinear')
        self.add_block_list(LayerType.Upsample, upsample3, self.block_out_channels[-1])

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
            elif LayerType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
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
