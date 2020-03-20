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
from easyai.model.base_block.seg.refinenet_block import RefineNetBlockName
from easyai.model.base_block.seg.refinenet_block import CRPBlock, RefineNetBlock
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class RefineNetSeg(BaseModel):

    def __init__(self, data_channel=3, class_num=2):
        super().__init__()
        self.set_name(ModelName.RefineNetSeg)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        backbone = self.factory.get_base_model(BackboneName.ResNet101)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        dropout1 = nn.Dropout(p=0.5)
        self.add_block_list(LayerType.Dropout, dropout1, self.block_out_channels[-1])

        layer1 = RouteLayer('31')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        dropout2 = nn.Dropout(p=0.5)
        self.add_block_list(LayerType.Dropout, dropout2, self.block_out_channels[-1])

        layer2 = RouteLayer('-3')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        conv1 = nn.Conv2d(2048, 512, 1, bias=False)
        self.add_block_list(LayerType.Convolutional, conv1, 512)

        layer3 = RouteLayer('-3')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer3.get_name(), layer3, output_channel)

        conv2 = nn.Conv2d(1024, 256, 1, bias=False)
        self.add_block_list(LayerType.Convolutional, conv2, 256)

        refinenet1 = RefineNetBlock(512, 512, 4, activation_name=self.activation_name)
        output_channel = 256
        self.add_block_list(refinenet1.get_name(), refinenet1, output_channel)

        layer4 = RouteLayer('8')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer4.get_name(), layer4, output_channel)

        conv3 = nn.Conv2d(512, 256, 1, bias=False)
        self.add_block_list(LayerType.Convolutional, conv3, 256)

        refinenet2 = RefineNetBlock(256, 256, 4, activation_name=self.activation_name)
        output_channel = 256
        self.add_block_list(refinenet2.get_name(), refinenet2, output_channel)

        layer5 = RouteLayer('4')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer5.get_name(), layer5, output_channel)

        conv4 = nn.Conv2d(256, 256, 1, bias=False)
        self.add_block_list(LayerType.Convolutional, conv4, 256)

        refinenet3 = RefineNetBlock(256, 256, 4, activation_name=self.activation_name)
        output_channel = 256
        self.add_block_list(refinenet3.get_name(), refinenet3, output_channel)

        crp = CRPBlock(256, 256, 4)
        output_channel = 256
        self.add_block_list(crp.get_name(), crp, output_channel)

        conv5 = nn.Conv2d(256, self.class_number, kernel_size=3, stride=1,
                          padding=1, bias=True)
        self.add_block_list(LayerType.Convolutional, conv5, self.class_number)

        layer6 = Upsample(scale_factor=4, mode='bilinear')
        self.add_block_list(layer6.get_name(), layer6, self.block_out_channels[-1])

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
            elif RefineNetBlockName.RefineNetBlock in key:
                x = block(layer_outputs[-3], layer_outputs[-1])
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output

