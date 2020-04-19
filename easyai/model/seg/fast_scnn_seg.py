#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""Fast Segmentation Convolutional Neural Network"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.separable_conv_block import SeparableConv2dBNActivation
from easyai.model.base_block.seg.fast_scnn_block import FastSCNNBlockName
from easyai.model.base_block.seg.fast_scnn_block import GlobalFeatureExtractor, FeatureFusionBlock
from easyai.model.utility.base_model import *


class FastSCNN(BaseModel):

    def __init__(self, data_channel=3, class_num=2):
        super().__init__()
        self.set_name(ModelName.FastSCNN)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.first_output = 64
        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        self.learning_to_downsample()

        global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128],
                                                          128, 6, [3, 3, 3],
                                                          bn_name=self.bn_name,
                                                          activation_name=self.activation_name)
        self.add_block_list(global_feature_extractor.get_name(), global_feature_extractor, 128)

        feature_fusion = FeatureFusionBlock(64, 128, 128,
                                            bn_name=self.bn_name,
                                            activation_name=self.activation_name)
        self.add_block_list(feature_fusion.get_name(), feature_fusion, 128)

        self.classifer(128)

        layer = Upsample(scale_factor=8, mode='bilinear')
        self.add_block_list(layer.get_name(), layer, self.block_out_channels[-1])

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = CrossEntropy2d(ignore_index=250)
        self.add_block_list(LossType.CrossEntropy2d, loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def learning_to_downsample(self):
        conv = ConvBNActivationBlock(in_channels=self.data_channel,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     bias=False,
                                     bnName=self.bn_name,
                                     activationName=self.activation_name)
        self.add_block_list(conv.get_name(), conv, 32)

        dsconv1 = SeparableConv2dBNActivation(32, 48, stride=2, relu_first=False,
                                              bn_name=self.bn_name,
                                              activation_name=self.activation_name)
        self.add_block_list(dsconv1.get_name(), dsconv1, 48)

        dsconv2 = SeparableConv2dBNActivation(48, self.first_output, stride=2, relu_first=False,
                                              bn_name=self.bn_name,
                                              activation_name=self.activation_name)
        self.add_block_list(dsconv2.get_name(), dsconv2, self.first_output)

    def classifer(self, dw_channels):
        dsconv1 = SeparableConv2dBNActivation(dw_channels, dw_channels, stride=1, relu_first=False,
                                              bn_name=self.bn_name,
                                              activation_name=self.activation_name)
        self.add_block_list(dsconv1.get_name(), dsconv1, dw_channels)

        dsconv2 = SeparableConv2dBNActivation(dw_channels, dw_channels, stride=1, relu_first=False,
                                              bn_name=self.bn_name,
                                              activation_name=self.activation_name)
        self.add_block_list(dsconv2.get_name(), dsconv2, dw_channels)

        dropout = nn.Dropout2d(0.1)
        self.add_block_list(LayerType.Dropout, dropout, self.block_out_channels[-1])

        conv = nn.Conv2d(dw_channels, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional, conv, self.class_number)

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
            elif FastSCNNBlockName.FeatureFusionBlock in key:
                x = block(layer_outputs[-2], layer_outputs[-1])
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        return output
