#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
title={Gated-SCNN: Gated Shape CNNs for Semantic Segmentation},
author={Takikawa, Towaki and Acuna, David and Jampani, Varun and Fidler, Sanja},
journal={ICCV},
year={2019}
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
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


class GSCNNSeg(BaseModel):

    def __init__(self, data_channel=3, class_num=19):
        super().__init__()
        self.set_name(ModelName.EncNetSeg)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.factory.get_base_model(BackboneName.wider_resnet38_a2)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])