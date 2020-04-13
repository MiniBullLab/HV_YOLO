#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.utility.create_model_list import CreateModuleList


class MyBackbone(BaseBackbone):

    def __init__(self, modelDefine):
        super().__init__()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        outChannels = []
        self.createTaskList.createOrderedDict(self.modelDefine, outChannels)
        blockDict = self.createTaskList.getBlockList()

        self.block_out_channels = self.createTaskList.getOutChannelList()
        for key, block in blockDict.items():
            name = "base_%s" % key
            self.add_module(name, block)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        for key, block in self._modules.items():
            if LayerType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortRouteLayer in key:
                x = block(layer_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            else:
                x = block(x)
            layer_outputs.append(x)
        return layer_outputs
