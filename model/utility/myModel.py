#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from collections import OrderedDict
from base_model.utility.base_model import *
from base_name.block_name import BlockType
from base_name.loss_name import LossType
from base_model.utility.base_model_factory import BaseModelFactory
from model.utility.createModelList import CreateModuleList


class MyModel(BaseModel):

    def __init__(self, modelDefine, cfg_dir):
        super().__init__()
        self.baseModelFactory = BaseModelFactory()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.cfg_dir = cfg_dir
        self.lossList = []
        self.createModel()

    def createModel(self):
        basicModel = self.creatBaseModel()
        self.add_module(BlockType.BaseNet, basicModel)
        taskBlockDict = self.createTask(basicModel)
        for key, block in taskBlockDict.items():
            self.add_module(key, block)
        self.lossList = self.createLoss(taskBlockDict)

    def creatBaseModel(self):
        result = None
        base_net = self.modelDefine[0]
        if base_net["type"] == BlockType.BaseNet:
            base_net_name = base_net["name"]
            self.modelDefine.pop(0)
            input_name = base_net_name.strip()
            if input_name.endswith("cfg"):
                input_name = os.path.join(self.cfg_dir, input_name)
            result = self.baseModelFactory.get_base_model(input_name)
        return result

    def createTask(self, basicModel):
        blockDict = OrderedDict()
        outChannels = basicModel.getOutChannelList()
        if basicModel:
            self.createTaskList.createOrderedDict(self.modelDefine, outChannels)
            blockDict = self.createTaskList.getBlockList()
        return blockDict

    def createLoss(self, taskBlockDict):
        lossResult = []
        for key, block in taskBlockDict.items():
            if LossType.CrossEntropy2d in key:
                lossResult.append(block)
            elif LossType.OhemCrossEntropy2d in key:
                lossResult.append(block)
            elif LossType.BinaryCrossEntropy2d in key:
                lossResult.append(block)
            elif LossType.YoloLoss in key:
                lossResult.append(block)
        return lossResult

    def setFreezeBn(self, freezeBn):
        for m in self.modules():
            if freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.ShortRouteLayer in key:
                x = block(layer_outputs)
            elif BlockType.ShortcutLayer in key:
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
