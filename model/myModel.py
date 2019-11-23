#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from collections import OrderedDict
from base_model.base_model import *
from base_name.block_name import BlockType
from base_block.darknet_block import DarknetBlockName
from base_name.loss_name import LossType
from base_model.base_model_factory import BaseModelFactory
from model.createModelList import CreateModuleList


class MyModel(BaseModel):

    def __init__(self, modelDefine):
        super().__init__()
        self.baseModelFactory = BaseModelFactory()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
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
        baseNet = self.modelDefine[0]
        if baseNet["type"] == BlockType.BaseNet:
            baseNetName = baseNet["name"]
            self.modelDefine.pop(0)
            result = self.baseModelFactory.get_base_model(baseNetName)
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
            print(x.shape)
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.Convolutional in key:
                x = block(x)
            elif BlockType.ConvActivationBlock in key:
                x = block(x)
            elif BlockType.ConvBNActivationBlock in key:
                x = block(x)
            elif BlockType.Upsample in key:
                x = block(x)
            elif BlockType.MyMaxPool2d in key:
                x = block(x)
            elif BlockType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif BlockType.GlobalAvgPool in key:
                x = block(x)
            elif BlockType.FcLayer in key:
                x = block(x)
            elif DarknetBlockName in key:
                x = block(x)
            elif LossType.YoloLoss in key:
                output.append(x)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            elif LossType.OhemCrossEntropy2d in key:
                output.append(x)
            print(key, x.shape)
            layer_outputs.append(x)
        return output
