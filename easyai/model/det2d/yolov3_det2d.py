#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.det2d.yolov3_loss import YoloV3Loss
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock, ConvActivationBlock
from easyai.model.utility.base_det_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class YoloV3Det2d(BaseDetectionModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.YoloV3Det2d)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.LeakyReLU

        self.anchor_sizes = [[10, 13], [16, 30], [33, 23],
                             [30, 61], [62, 45], [59, 119],
                             [116, 90], [156, 198], [373, 326]]

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()
        self.lossList = []

        backbone = self.factory.get_base_model(BackboneName.Darknet53, self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = ConvBNActivationBlock(in_channels=base_out_channels[-1],
                                       out_channels=512,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, 512)

        layer2 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=1024,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, 1024)

        layer3 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=512,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer3.get_name(), layer3, 512)

        layer4 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=1024,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer4.get_name(), layer4, 1024)

        layer5 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=512,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer5.get_name(), layer5, 512)

        layer6 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=1024,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer6.get_name(), layer6, 1024)

        output_filter = 3 * (4 + 1 + self.class_number)
        layer7 = ConvActivationBlock(in_channels=self.block_out_channels[-1],
                                     out_channels=output_filter,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     activationName=ActivationType.Linear)
        self.add_block_list(layer7.get_name(), layer7, output_filter)

        loss1 = YoloV3Loss(self.class_number, self.anchor_sizes,
                           anchor_mask=(6, 7, 8), reduction=32)
        self.add_block_list(loss1.get_name(), loss1, output_filter)
        self.lossList.append(loss1)

        route1 = RouteLayer('-4')
        output_channel = route1.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route1.get_name(), route1, output_channel)

        layer8 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=256,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer8.get_name(), layer8, 256)

        up1 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(up1.get_name(), up1, self.block_out_channels[-1])

        route2 = RouteLayer('-1,23')
        output_channel = route2.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route2.get_name(), route2, output_channel)

        layer9 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=256,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer9.get_name(), layer9, 256)

        layer10 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=512,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer10.get_name(), layer10, 512)

        layer11 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer11.get_name(), layer11, 256)

        layer12 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=512,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer12.get_name(), layer12, 512)

        layer13 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer13.get_name(), layer13, 256)

        layer14 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=512,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer14.get_name(), layer14, 512)

        output_filter = 3 * (4 + 1 + self.class_number)
        layer15 = ConvActivationBlock(in_channels=self.block_out_channels[-1],
                                      out_channels=output_filter,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      activationName=ActivationType.Linear)
        self.add_block_list(layer15.get_name(), layer15, output_filter)

        loss2 = YoloV3Loss(self.class_number, self.anchor_sizes,
                           anchor_mask=(3, 4, 5), reduction=16)
        self.add_block_list(loss2.get_name(), loss2, output_filter)
        self.lossList.append(loss2)

        route3 = RouteLayer('-4')
        output_channel = route3.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route3.get_name(), route3, output_channel)

        layer16 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer16.get_name(), layer16, 128)

        up2 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(up2.get_name(), up2, self.block_out_channels[-1])

        route4 = RouteLayer('-1,14')
        output_channel = route4.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route4.get_name(), route4, output_channel)

        layer17 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer17.get_name(), layer17, 128)

        layer18 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer18.get_name(), layer18, 256)

        layer19 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer19.get_name(), layer19, 128)

        layer20 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer20.get_name(), layer20, 256)

        layer21 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer21.get_name(), layer21, 128)

        layer22 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer22.get_name(), layer22, 256)

        output_filter = 3 * (4 + 1 + self.class_number)
        layer23 = ConvActivationBlock(in_channels=self.block_out_channels[-1],
                                      out_channels=output_filter,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      activationName=ActivationType.Linear)
        self.add_block_list(layer23.get_name(), layer23, output_filter)

        loss3 = YoloV3Loss(self.class_number, self.anchor_sizes,
                           anchor_mask=(0, 1, 2), reduction=8)
        self.add_block_list(loss3.get_name(), loss3, output_filter)
        self.lossList.append(loss3)

        self.create_loss()

    def create_loss(self, input_dict=None):
        pass

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
            elif LossType.YoloV3Loss in key:
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
