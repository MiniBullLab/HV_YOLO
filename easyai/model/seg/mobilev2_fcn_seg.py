#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.base_name.backbone_name import BackboneName
from easyai.loss.cls.ce2d_loss import CrossEntropy2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.MobileV2FCN)
class MobileV2FCN(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=2):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.MobileV2FCN)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU6

        self.model_args['type'] = BackboneName.MobileNetV2_1_0

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        backbone = self.factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        input_channel = self.block_out_channels[-1]
        output_channel = base_out_channels[-1] // 2
        self.make_conv_block(input_channel, output_channel)

        self.make_layer(base_out_channels, output_channel, 2, '13')

        input_channel = self.block_out_channels[-1]
        output_channel = base_out_channels[-1] // 4
        self.make_conv_block(input_channel, output_channel)

        self.make_layer(base_out_channels, output_channel, 2, '6')

        input_channel = self.block_out_channels[-1]
        output_channel = base_out_channels[-1] // 8
        self.make_conv_block(input_channel, output_channel)

        self.make_layer(base_out_channels, output_channel, 2, '3')

        input_channel = self.block_out_channels[-1]
        output_channel = base_out_channels[-1] // 16
        self.make_conv_block(input_channel, output_channel)

        input_channel = self.block_out_channels[-1]
        output_channel = self.class_number
        conv4 = ConvActivationBlock(input_channel, output_channel,
                                    kernel_size=1, stride=1, padding=0,
                                    activationName=ActivationType.Linear)
        self.add_block_list(conv4.get_name(), conv4, output_channel)

        layer10 = Upsample(scale_factor=4, mode='bilinear')
        self.add_block_list(layer10.get_name(), layer10, self.block_out_channels[-1])

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = CrossEntropy2d(ignore_index=250)
        self.add_block_list(LossType.CrossEntropy2d, loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def make_layer(self, base_out_channels, conv_output_channel, scale_factor, route_layer_indexs):
        layer1 = Upsample(scale_factor=scale_factor, mode='bilinear')
        self.add_block_list(layer1.get_name(), layer1, self.block_out_channels[-1])

        layer2 = RouteLayer(route_layer_indexs)
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        conv1 = ConvBNActivationBlock(self.block_out_channels[-1], conv_output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, conv_output_channel)

        layer3 = RouteLayer('-1,-3')
        output_channel = sum([base_out_channels[i] if i >= 0 else
                              self.block_out_channels[i] for i in layer3.layers])
        self.add_block_list(layer3.get_name(), layer3, output_channel)

    def make_conv_block(self, input_channel, output_channel):
        conv1 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, output_channel)

        temp_input_channel = self.block_out_channels[-1]
        temp_output_channel = output_channel * 2
        conv2 = ConvBNActivationBlock(temp_input_channel, temp_output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, temp_output_channel)

        temp_input_channel = self.block_out_channels[-1]
        temp_output_channel = output_channel
        conv3 = ConvBNActivationBlock(temp_input_channel, temp_output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, temp_output_channel)

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
