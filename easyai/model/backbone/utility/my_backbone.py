#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import LayerType
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.utility.create_model_list import CreateModuleList


class MyBackbone(BaseBackbone):

    def __init__(self, model_defines):
        super().__init__(None)
        self.createTaskList = CreateModuleList()
        self.model_defines = model_defines
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        base_out_channels = []
        self.createTaskList.createOrderedDict(self.model_defines, base_out_channels)
        block_dict = self.createTaskList.getBlockList()

        out_channel_list = self.createTaskList.getOutChannelList()
        for index, (key, block) in enumerate(block_dict.items()):
            name = "base_%s" % key
            self.add_block_list(name, block, out_channel_list[index], flag=1)

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
