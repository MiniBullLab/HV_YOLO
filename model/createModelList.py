import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from collections import OrderedDict
import torch.nn as nn
from base_block.blockName import BlockType, LossType
from base_block.utilityBlock import ConvBNActivationBlock, ConvActivationBlock
from base_block.baseLayer import EmptyLayer, RouteLayer, ShortcutLayer, Upsample, GlobalAvgPool2d
from loss import *

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
        for module_def in modelDefine:
            if module_def['type'] == BlockType.Convolutional:
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                block = nn.Conv2d(in_channels=self.outChannelList[-1],
                                out_channels=filters,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=pad,
                                bias=True)
                self.addBlockList(BlockType.Convolutional, block, filters)
            elif module_def['type'] == BlockType.ConvActivationBlock:
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                activationName = module_def['activation']
                block = ConvActivationBlock(in_channels=self.outChannelList[-1],
                                          out_channels=filters,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=pad,
                                          activationName=activationName)
                self.addBlockList(BlockType.ConvActivationBlock, block, filters)
            elif module_def['type'] == BlockType.ConvBNActivationBlock:
                bnName = module_def['batch_normalize']
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                activationName = module_def['activation']
                block = ConvBNActivationBlock(in_channels=self.outChannelList[-1],
                                              out_channels=filters,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=pad,
                                              bnName=bnName,
                                              activationName=activationName)
                self.addBlockList(BlockType.ConvBNActivationBlock, block, filters)
            elif module_def['type'] == BlockType.Maxpool:
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    block = nn.ZeroPad2d((0, 1, 0, 1))
                    self.addBlockList(BlockType.Maxpool, block, filters)
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                self.addBlockList(BlockType.Maxpool, maxpool, filters)
            elif module_def['type'] == BlockType.GlobalAvgPool:
                globalAvgPool = GlobalAvgPool2d()
                self.addBlockList(BlockType.GlobalAvgPool, globalAvgPool, filters)
            elif module_def['type'] == BlockType.Upsample:
                upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
                self.addBlockList(BlockType.Upsample, upsample, filters)
            elif module_def['type'] == BlockType.RouteLayer:
                block = RouteLayer(module_def['layers'])
                filters = sum([inputChannels[i] if i >= 0 else self.outChannelList[i] for i in block.layers])
                self.addBlockList(BlockType.RouteLayer, block, filters)
            elif module_def['type'] == BlockType.ShortcutLayer:
                block = ShortcutLayer(module_def['from'])
                filters = self.outChannelList[block.layer_from]
                self.addBlockList(BlockType.ShortcutLayer, block, filters)
            elif module_def['type'] == LossType.Yolo:
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                num_classes = int(module_def['classes'])
                # Define detection layer
                yolo_layer = YoloLoss(num_classes, anchors=anchors, anchors_mask=anchor_idxs, smoothLabel=False,
                                      focalLoss=False)
                self.addBlockList(LossType.Yolo, yolo_layer, filters)
            elif module_def["type"] == LossType.Softmax:
                self.addBlockList(LossType.Softmax, EmptyLayer(), filters)

    def addBlockList(self, blockName, block, out_channel):
        blockName = "%s_%d" % (blockName, self.index)
        self.blockDict[blockName] = block
        self.outChannelList.append(out_channel)
        self.index += 1