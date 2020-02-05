#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from collections import OrderedDict
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.create_model_list import CreateModuleList
from easyai.model.utility.base_model import *


class MyModel(BaseModel):

    def __init__(self, modelDefine, cfg_dir):
        super().__init__()
        self.backbone_factory = BackboneFactory()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.cfg_dir = cfg_dir

        self.create_block_list()

    def create_block_list(self):
        basicModel = self.creatBaseModel()
        self.add_module(BlockType.BaseNet, basicModel)
        taskBlockDict = self.createTask(basicModel)
        for key, block in taskBlockDict.items():
            self.add_module(key, block)
        self.create_loss(input_dict=taskBlockDict)

    def create_loss(self, input_dict=None):
        self.lossList = []
        for key, block in input_dict.items():
            if LossType.CrossEntropy2d in key:
                self.lossList.append(block)
            elif LossType.OhemCrossEntropy2d in key:
                self.lossList.append(block)
            elif LossType.BinaryCrossEntropy2d in key:
                self.lossList.append(block)
            elif LossType.YoloLoss in key:
                self.lossList.append(block)

    def creatBaseModel(self):
        result = None
        base_net = self.modelDefine[0]
        if base_net["type"] == BlockType.BaseNet:
            base_net_name = base_net["name"]
            self.modelDefine.pop(0)
            input_name = base_net_name.strip()
            if input_name.endswith("cfg"):
                input_name = os.path.join(self.cfg_dir, input_name)
            result = self.backbone_factory.get_base_model(input_name)
        return result

    def createTask(self, basicModel):
        blockDict = OrderedDict()
        outChannels = basicModel.get_outchannel_list()
        if basicModel:
            self.createTaskList.createOrderedDict(self.modelDefine, outChannels)
            blockDict = self.createTaskList.getBlockList()
        return blockDict

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
            elif LayerType.ShortRouteLayer in key:
                x = block(layer_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif LossType.YoloLoss in key:
                output.append(x)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            elif LossType.OhemCrossEntropy2d in key:
                output.append(x)
            elif LossType.BinaryCrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
