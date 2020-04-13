#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.nn as nn
from collections import OrderedDict
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock, ConvActivationBlock
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility.utility_layer import MultiplyLayer, AddLayer
from easyai.model.base_block.utility.utility_layer import RouteLayer, ShortRouteLayer
from easyai.model.base_block.utility.utility_layer import ShortcutLayer
from easyai.model.base_block.utility.utility_layer import FcLayer
from easyai.model.base_block.utility.pooling_layer import MyMaxPool2d, GlobalAvgPool2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.cls.darknet_block import ReorgBlock, DarknetBlockName
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.loss.utility.bce_loss import BinaryCrossEntropy2d
from easyai.loss.seg.ohem_cross_entropy2d import OhemCrossEntropy2d
from easyai.loss.det2d.yolo_loss import YoloLoss


class CreateModuleList():

    def __init__(self):
        self.index = 0
        self.outChannelList = []
        self.blockDict = OrderedDict()

        self.filters = 0
        self.input_channels = 0

    def getBlockList(self):
        return self.blockDict

    def getOutChannelList(self):
        return self.outChannelList

    def createOrderedDict(self, modelDefine, inputChannels):
        self.index = 0
        self.blockDict.clear()
        self.outChannelList.clear()
        self.filters = 0
        self.input_channels = 0

        for module_def in modelDefine:
            if module_def['type'] == BlockType.InputData:
                data_channel = int(module_def['data_channel'])
                self.input_channels = data_channel
            elif module_def['type'] == LayerType.RouteLayer:
                block = RouteLayer(module_def['layers'])
                self.filters = sum([inputChannels[i] if i >= 0 else self.outChannelList[i]
                                    for i in block.layers])
                self.addBlockList(LayerType.RouteLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == LayerType.ShortRouteLayer:
                block = ShortRouteLayer(module_def['from'], module_def['activation'])
                self.filters = self.outChannelList[block.layer_from] + \
                               self.outChannelList[-1]
                self.addBlockList(LayerType.ShortRouteLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == LayerType.ShortcutLayer:
                block = ShortcutLayer(module_def['from'], module_def['activation'])
                self.filters = self.outChannelList[block.layer_from]
                self.addBlockList(LayerType.ShortcutLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == DarknetBlockName.ReorgBlock:
                stride = int(module_def['stride'])
                block = ReorgBlock(stride=stride)
                self.filters = block.stride * block.stride * self.outChannelList[-1]
                self.addBlockList(DarknetBlockName.ReorgBlock, block, self.filters)
                self.input_channels = self.filters
            else:
                self.create_layer(module_def)
                self.create_convolutional(module_def)
                self.create_loss(module_def)

    def create_layer(self, module_def):
        if module_def['type'] == LayerType.MyMaxPool2d:
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            maxpool = MyMaxPool2d(kernel_size, stride)
            self.addBlockList(LayerType.MyMaxPool2d, maxpool, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.GlobalAvgPool:
            globalAvgPool = GlobalAvgPool2d()
            self.addBlockList(LayerType.GlobalAvgPool, globalAvgPool, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.FcLayer:
            num_output = int(module_def['num_output'])
            self.filters = num_output
            layer = FcLayer(self.input_channels, num_output)
            self.addBlockList(LayerType.FcLayer, layer, num_output)
            self.input_channels = num_output
        elif module_def['type'] == LayerType.Upsample:
            scale = int(module_def['stride'])
            mode = module_def.get('model', 'bilinear')
            upsample = Upsample(scale_factor=scale, mode=mode)
            self.addBlockList(LayerType.Upsample, upsample, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.MultiplyLayer:
            layer = MultiplyLayer(module_def['layers'])
            self.addBlockList(LayerType.MultiplyLayer, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.AddLayer:
            layer = AddLayer(module_def['layers'])
            self.addBlockList(LayerType.AddLayer, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.Dropout:
            probability = float(module_def['probability'])
            layer = nn.Dropout(p=probability, inplace=False)
            self.addBlockList(LayerType.Dropout, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.NormalizeLayer:
            bn_name = module_def['batch_normalize'].strip()
            layer = NormalizeLayer(bn_name, self.filters)
            self.addBlockList(LayerType.NormalizeLayer, layer, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == LayerType.ActivationLayer:
            activation_name = module_def['activation'].strip()
            layer = ActivationLayer(activation_name, inplace=False)
            self.addBlockList(LayerType.ActivationLayer, layer, self.filters)
            self.input_channels = self.filters

    def create_convolutional(self, module_def):
        if module_def['type'] == LayerType.Convolutional:
            self.filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            dilation = int(module_def.get('dilation', 1))
            if dilation > 1:
                pad = dilation
            block = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=True)
            self.addBlockList(LayerType.Convolutional, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == BlockType.ConvActivationBlock:
            self.filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            activationName = module_def['activation']
            dilation = int(module_def.get("dilation", 1))
            if dilation > 1:
                pad = dilation
            block = ConvActivationBlock(in_channels=self.input_channels,
                                        out_channels=self.filters,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=pad,
                                        dilation=dilation,
                                        activationName=activationName)
            self.addBlockList(BlockType.ConvActivationBlock, block, self.filters)
            self.input_channels = self.filters
        elif module_def['type'] == BlockType.ConvBNActivationBlock:
            bnName = module_def['batch_normalize']
            self.filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            activationName = module_def['activation']
            dilation = int(module_def.get("dilation", 1))
            if dilation > 1:
                pad = dilation
            block = ConvBNActivationBlock(in_channels=self.input_channels,
                                          out_channels=self.filters,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=pad,
                                          bnName=bnName,
                                          dilation=dilation,
                                          activationName=activationName)
            self.addBlockList(BlockType.ConvBNActivationBlock, block, self.filters)
            self.input_channels = self.filters

    def create_loss(self, module_def):
        if module_def['type'] == LossType.YoloLoss:
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            num_classes = int(module_def['classes'])
            # Define detection layer
            yolo_layer = YoloLoss(num_classes, anchors=anchors,
                                  anchors_mask=anchor_idxs, smoothLabel=False,
                                  focalLoss=False)
            self.addBlockList(LossType.YoloLoss, yolo_layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.CrossEntropy2d:
            weight_type = int(module_def.get("weight_type", 0))
            weight = module_def.get("weight", None)
            reduce = module_def.get("reduce", None)
            reduction = module_def.get("reduction", 'mean')
            ignore_index = int(module_def.get("ignore_index", 250))
            layer = CrossEntropy2d(weight_type=weight_type,
                                   weight=weight,
                                   reduce=reduce,
                                   reduction=reduction,
                                   ignore_index=ignore_index)
            self.addBlockList(LossType.CrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.OhemCrossEntropy2d:
            ignore_index = int(module_def.get("ignore_index", 250))
            layer = OhemCrossEntropy2d(ignore_index=ignore_index)
            self.addBlockList(LossType.OhemCrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.BinaryCrossEntropy2d:
            weight_type = int(module_def.get("weight_type", 0))
            weight = module_def.get("weight", None)
            reduce = module_def.get("reduce", None)
            reduction = module_def.get("reduction", 'mean')
            layer = BinaryCrossEntropy2d(weight_type=weight_type,
                                         weight=weight,
                                         reduce=reduce,
                                         reduction=reduction)
            self.addBlockList(LossType.BinaryCrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters

    def addBlockList(self, blockName, block, out_channel):
        blockName = "%s_%d" % (blockName, self.index)
        self.blockDict[blockName] = block
        self.outChannelList.append(out_channel)
        self.index += 1
