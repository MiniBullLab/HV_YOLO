#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.cls.ce2d_loss import CrossEntropy2d
from easyai.model.base_block.utility.utility_layer import FcLayer
from easyai.model.base_block.utility.pooling_layer import GlobalAvgPool2d
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.registry import REGISTERED_CLS_MODEL


@REGISTERED_CLS_MODEL.register_module(ModelName.ClassNet)
class ClassNet(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=100):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.ClassNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.ResNet18

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = ConvBNActivationBlock(in_channels=base_out_channels[-1],
                                       out_channels=1280,
                                       kernel_size=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, 1280)

        avgpool = GlobalAvgPool2d()
        self.add_block_list(avgpool.get_name(), avgpool, self.block_out_channels[-1])

        layer1 = FcLayer(self.block_out_channels[-1], self.class_number)
        self.add_block_list(layer1.get_name(), layer1, self.class_number)

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = CrossEntropy2d(reduction='mean', ignore_index=250)
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
