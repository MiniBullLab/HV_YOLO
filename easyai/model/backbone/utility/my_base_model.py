#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import BlockType
from easyai.model.backbone.utility.base_model import *
from easyai.model.utility.create_model_list import CreateModuleList


class MyBaseModel(BaseModel):

    def __init__(self, modelDefine):
        super().__init__()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.outChannelList = []
        self.createModel()

    def createModel(self):
        outChannels = []
        self.createTaskList.createOrderedDict(self.modelDefine, outChannels)
        blockDict = self.createTaskList.getBlockList()
        self.outChannelList = self.createTaskList.getOutChannelList()
        for key, block in blockDict.items():
            name = "base_%s" % key
            self.add_module(name, block)

    def getOutChannelList(self):
        return self.outChannelList

    def setFreezeBn(self, freezeBn):
        for m in self.modules():
            if freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def printBlockName(self):
        for key in self._modules.keys():
            print(key)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        for key, block in self._modules.items():
            if BlockType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.ShortRouteLayer in key:
                x = block(layer_outputs)
            elif BlockType.ShortcutLayer in key:
                x = block(layer_outputs)
            else:
                x = block(x)
            layer_outputs.append(x)
        return layer_outputs
