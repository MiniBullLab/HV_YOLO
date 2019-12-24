#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.nn as nn
from collections import OrderedDict
from base_name.block_name import BlockType
from base_name.loss_name import LossType
from base_block.utility_block import ConvBNActivationBlock, ConvActivationBlock
from base_block.utility_block import BNActivationBlock
from base_block.utility_layer import MultiplyLayer, AddLayer
from base_block.utility_layer import RouteLayer, ShortRouteLayer
from base_block.utility_layer import ShortcutLayer
from base_block.utility_layer import MyMaxPool2d, Upsample, GlobalAvgPool2d, FcLayer
from base_block.darknet_block import ReorgBlock, DarknetBlockName
from loss.cross_entropy2d import CrossEntropy2d
from loss.bce_loss import BinaryCrossEntropy2d
from loss.ohem_cross_entropy2d import OhemCrossEntropy2d
from loss.yolo_loss import YoloLoss


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
            elif module_def['type'] == BlockType.MyMaxPool2d:
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                maxpool = MyMaxPool2d(kernel_size, stride)
                self.addBlockList(BlockType.MyMaxPool2d, maxpool, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.GlobalAvgPool:
                globalAvgPool = GlobalAvgPool2d()
                self.addBlockList(BlockType.GlobalAvgPool, globalAvgPool, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.FcLayer:
                num_output = int(module_def['num_output'])
                self.filters = num_output
                block = FcLayer(self.input_channels, num_output)
                self.addBlockList(BlockType.FcLayer, block, num_output)
                self.input_channels = num_output
            elif module_def['type'] == BlockType.Upsample:
                upsample = Upsample(scale_factor=int(module_def['stride']))
                self.addBlockList(BlockType.Upsample, upsample, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.RouteLayer:
                block = RouteLayer(module_def['layers'])
                self.filters = sum([inputChannels[i] if i >= 0 else self.outChannelList[i]
                                    for i in block.layers])
                self.addBlockList(BlockType.RouteLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.ShortRouteLayer:
                block = ShortRouteLayer(module_def['from'], module_def['activation'])
                self.filters = self.outChannelList[block.layer_from] + \
                               self.outChannelList[-1]
                self.addBlockList(BlockType.ShortRouteLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.ShortcutLayer:
                block = ShortcutLayer(module_def['from'], module_def['activation'])
                self.filters = self.outChannelList[block.layer_from]
                self.addBlockList(BlockType.ShortcutLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.MultiplyLayer:
                block = MultiplyLayer(module_def['layers'])
                self.addBlockList(BlockType.MultiplyLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.AddLayer:
                block = AddLayer(module_def['layers'])
                self.addBlockList(BlockType.AddLayer, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == BlockType.Dropout:
                probability = float(module_def['probability'])
                block = nn.Dropout(p=probability, inplace=True)
                self.addBlockList(BlockType.Dropout, block, self.filters)
                self.input_channels = self.filters
            elif module_def['type'] == DarknetBlockName.ReorgBlock:
                stride = int(module_def['stride'])
                block = ReorgBlock(stride=stride)
                self.filters = block.stride * block.stride * self.outChannelList[-1]
                self.addBlockList(DarknetBlockName.ReorgBlock, block, self.filters)
                self.input_channels = self.filters
            else:
                self.create_convolutional(module_def)
                self.create_loss(module_def)

    def create_convolutional(self, module_def):
        if module_def['type'] == BlockType.Convolutional:
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
            self.addBlockList(BlockType.Convolutional, block, self.filters)
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
        elif module_def['type'] == BlockType.BNActivationBlock:
            bnName = module_def['batch_normalize']
            activationName = module_def['activation']
            block = BNActivationBlock(in_channels=self.filters,
                                      bnName=bnName,
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
            ignore_index = int(module_def["ignore_index"])
            layer = CrossEntropy2d(ignore_index=ignore_index)
            self.addBlockList(LossType.CrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.OhemCrossEntropy2d:
            ignore_index = int(module_def["ignore_index"])
            layer = OhemCrossEntropy2d(ignore_index=ignore_index)
            self.addBlockList(LossType.OhemCrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters
        elif module_def["type"] == LossType.BinaryCrossEntropy2d:
            weight = [float(x) for x in module_def["weight"].split(',') if x]
            layer = BinaryCrossEntropy2d(weight=weight)
            self.addBlockList(LossType.BinaryCrossEntropy2d, layer, self.filters)
            self.input_channels = self.filters

    def addBlockList(self, blockName, block, out_channel):
        blockName = "%s_%d" % (blockName, self.index)
        self.blockDict[blockName] = block
        self.outChannelList.append(out_channel)
        self.index += 1