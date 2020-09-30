#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.create_model_list import CreateModuleList
from easyai.model.utility.base_model import *


class MyModel(BaseModel):

    def __init__(self, model_defines, cfg_dir, default_args=None):
        super().__init__(None)
        self.backbone_factory = BackboneFactory()
        self.createTaskList = CreateModuleList()
        self.model_defines = model_defines
        self.cfg_dir = cfg_dir
        self.default_args = default_args
        self.backbone_name = ""

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone_block, self.backbone_name = self.creat_backbone()
        base_out_channels = backbone_block.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone_block,
                            base_out_channels[-1])
        if backbone_block is not None:
            task_block_dict, task_out_channels = self.create_task(base_out_channels)
            for index, (key, block) in enumerate(task_block_dict.items()):
                self.add_block_list(key, block, task_out_channels[index], flag=1)

            self.create_loss(input_dict=task_block_dict)
        else:
            print("create backbone error!")

    def create_loss(self, input_dict=None):
        self.lossList = []
        for key, block in input_dict.items():
            if LossType.CrossEntropy2d in key:
                self.lossList.append(block)
            elif LossType.BinaryCrossEntropy2d in key:
                self.lossList.append(block)
            elif LossType.OhemCrossEntropy2d in key:
                self.lossList.append(block)
            elif LossType.Region2dLoss in key:
                self.lossList.append(block)
            elif LossType.YoloV3Loss in key:
                self.lossList.append(block)
            elif LossType.MultiBoxLoss in key:
                self.lossList.append(block)
            elif LossType.KeyPoints2dRegionLoss in key:
                self.lossList.append(block)

    def creat_backbone(self):
        input_name = ''
        result = None
        backbone = self.model_defines[0]
        if backbone["type"] == BlockType.BaseNet:
            input_name = backbone["name"]
            self.model_defines.pop(0)
            input_name = input_name.strip()
            if input_name.endswith("cfg"):
                input_cfg_path = os.path.join(self.cfg_dir, input_name)
                self.default_args['type'] = input_cfg_path
                result = self.backbone_factory.get_backbone_model(self.default_args)
            else:
                self.default_args['type'] = input_name
                result = self.backbone_factory.get_backbone_model(self.default_args)
        return result, input_name

    def create_task(self, base_out_channels):
        self.createTaskList.set_start_index(1)
        self.createTaskList.createOrderedDict(self.model_defines, base_out_channels)
        block_dict = self.createTaskList.getBlockList()
        task_out_channels = self.createTaskList.getOutChannelList()
        return block_dict, task_out_channels

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        multi_output = []
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
            elif LayerType.ShortRouteLayer in key:
                x = block(layer_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif BlockType.Detection2dBlock in key:
                x = block(x)
                multi_output.extend(x)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            elif LossType.BinaryCrossEntropy2d in key:
                output.append(x)
            elif LossType.OhemCrossEntropy2d in key:
                output.append(x)
            elif LossType.KeyPoints2dRegionLoss in key:
                output.append(x)
            elif LossType.Region2dLoss in key:
                output.append(x)
            elif LossType.YoloV3Loss in key:
                output.append(x)
            elif LossType.MultiBoxLoss in key:
                output.extend(multi_output)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
