#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.nn as nn
from collections import OrderedDict
from base_name.block_name import BlockType
from base_name.loss_name import LossType
from base_block.utility_block import ConvBNActivationBlock, ConvActivationBlock
from base_block.utility_layer import RouteLayer, ShortcutLayer
from base_block.utility_layer import MyMaxPool2d, Upsample, GlobalAvgPool2d, FcLayer
from loss.cross_entropy2d import CrossEntropy2d
from loss.ohem_cross_entropy2d import OhemCrossEntropy2d
from loss.yolo_loss import YoloLoss


class CreateModuleList():

    def __init__(self):
        self.index = 0
        self.outChannelList = []
        self.blockDict = OrderedDict()

    def getBlockList(self):
        return self.blockDict

    def getOutChannelList(self):
        return self.outChannelList

    def createOrderedDict(self, modelDefine, inputChannels):
        self.index = 0
        self.blockDict.clear()
        self.outChannelList.clear()
        input_channels = 0
        for module_def in modelDefine:
            if module_def['type'] == BlockType.InputData:
                data_channel = int(module_def['data_channel'])
                input_channels = data_channel
            if module_def['type'] == BlockType.Convolutional:
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                block = nn.Conv2d(in_channels=input_channels,
                                out_channels=filters,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=pad,
                                bias=True)
                self.addBlockList(BlockType.Convolutional, block, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.ConvActivationBlock:
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                activationName = module_def['activation']
                block = ConvActivationBlock(in_channels=input_channels,
                                          out_channels=filters,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=pad,
                                          activationName=activationName)
                self.addBlockList(BlockType.ConvActivationBlock, block, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.ConvBNActivationBlock:
                bnName = module_def['batch_normalize']
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                activationName = module_def['activation']
                block = ConvBNActivationBlock(in_channels=input_channels,
                                              out_channels=filters,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=pad,
                                              bnName=bnName,
                                              activationName=activationName)
                self.addBlockList(BlockType.ConvBNActivationBlock, block, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.MyMaxPool2d:
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                maxpool = MyMaxPool2d(kernel_size, stride)
                self.addBlockList(BlockType.MyMaxPool2d, maxpool, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.GlobalAvgPool:
                globalAvgPool = GlobalAvgPool2d()
                self.addBlockList(BlockType.GlobalAvgPool, globalAvgPool, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.FcLayer:
                num_output = int(module_def['num_output'])
                block = FcLayer(input_channels, num_output)
                self.addBlockList(BlockType.FcLayer, block, num_output)
                input_channels = num_output
            elif module_def['type'] == BlockType.Upsample:
                upsample = Upsample(scale_factor=int(module_def['stride']))
                self.addBlockList(BlockType.Upsample, upsample, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.RouteLayer:
                block = RouteLayer(module_def['layers'])
                filters = sum([inputChannels[i] if i >= 0 else self.outChannelList[i] for i in block.layers])
                self.addBlockList(BlockType.RouteLayer, block, filters)
                input_channels = filters
            elif module_def['type'] == BlockType.ShortcutLayer:
                block = ShortcutLayer(module_def['from'], module_def['activation'])
                filters = self.outChannelList[block.layer_from]
                self.addBlockList(BlockType.ShortcutLayer, block, filters)
                input_channels = filters
            elif module_def['type'] == LossType.YoloLoss:
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                num_classes = int(module_def['classes'])
                # Define detection layer
                yolo_layer = YoloLoss(num_classes, anchors=anchors, anchors_mask=anchor_idxs, smoothLabel=False,
                                      focalLoss=False)
                self.addBlockList(LossType.YoloLoss, yolo_layer, filters)
                input_channels = filters
            elif module_def["type"] == LossType.CrossEntropy2d:
                ignore_index = int(module_def["ignore_index"])
                layer = CrossEntropy2d(ignore_index=ignore_index)
                self.addBlockList(LossType.CrossEntropy2d, layer, filters)
                input_channels = filters
            elif module_def["type"] == LossType.OhemCrossEntropy2d:
                ignore_index = int(module_def["ignore_index"])
                layer = OhemCrossEntropy2d(ignore_index=ignore_index)
                self.addBlockList(LossType.OhemCrossEntropy2d, layer, filters)
                input_channels = filters

    def addBlockList(self, blockName, block, out_channel):
        blockName = "%s_%d" % (blockName, self.index)
        self.blockDict[blockName] = block
        self.outChannelList.append(out_channel)
        self.index += 1