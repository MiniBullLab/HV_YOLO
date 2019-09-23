import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch.nn as nn
from base_block.blockName import BlockType, LossType
from base_block.utilityBlock import ConvBNActivationBlock, ConvActivationBlock
from base_block.baseLayer import EmptyLayer, Upsample, GlobalAvgPool2d
from loss import *

class CreateModuleList():

    def __init__(self):
        pass

    def getModuleList(self, inputChannels, modelDefine):
        output_filters = [inputChannels[-1]]
        moduleList = nn.ModuleList()
        for i, module_def in enumerate(modelDefine):
            modules = nn.Sequential()

            if module_def['type'] == BlockType.Convolutional:
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                block = nn.Conv2d(in_channels=output_filters[-1],
                                out_channels=filters,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=pad,
                                bias=True)
                blockName = "%s_%d" % (BlockType.Convolutional, i)
                modules.add_module(blockName, block)
            elif module_def['type'] == BlockType.ConvActivationBlock:
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                activationName = module_def['activation']
                block = ConvActivationBlock(in_channels=output_filters[-1],
                                          out_channels=filters,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=pad,
                                          activationName=activationName)
                blockName = "%s_%d" % (BlockType.ConvActivationBlock, i)
                modules.add_module(blockName, block)
            elif module_def['type'] == BlockType.ConvBNActivationBlock:
                bnName = module_def['batch_normalize']
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                activationName = module_def['activation']
                block = ConvBNActivationBlock(in_channels=output_filters[-1],
                                              out_channels=filters,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=pad,
                                              bnName=bnName,
                                              activationName=activationName)
                blockName = "%s_%d" % (BlockType.ConvBNActivationBlock, i)
                modules.add_module(blockName, block)
            elif module_def['type'] == BlockType.Maxpool:
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                blockName = "%s_%d" % (BlockType.Maxpool, i)
                modules.add_module(blockName, maxpool)
            elif module_def['type'] == BlockType.GlobalAvgPool:
                globalAvgPool = GlobalAvgPool2d()
                blockName = "%s_%d" % (BlockType.GlobalAvgPool, i)
                modules.add_module(blockName, globalAvgPool)
            elif module_def['type'] == BlockType.Upsample:
                # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
                #upsample = Upsample(scale_factor=int(module_def['stride']), mode='bilinear')
                upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
                blockName = "%s_%d" % (BlockType.Upsample, i)
                modules.add_module(blockName, upsample)
            elif module_def['type'] == BlockType.Route:
                layers = [int(x) for x in module_def['layers'].split(',')]
                filters = sum([inputChannels[i] if i >= 0 else output_filters[i] for i in layers])
                # filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                blockName = "%s_%d" % (BlockType.Route, i)
                modules.add_module(blockName, EmptyLayer())
            elif module_def['type'] == BlockType.Shortcut:
                filters = output_filters[int(module_def['from'])]
                blockName = "%s_%d" % (BlockType.Shortcut, i)
                modules.add_module(blockName, EmptyLayer())
            elif module_def['type'] == LossType.Yolo:
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                num_classes = int(module_def['classes'])
                # Define detection layer
                yolo_layer = YoloLoss(num_classes, anchors=anchors, anchors_mask=anchor_idxs, smoothLabel=False,
                                      focalLoss=False)
                blockName = "%s_%d" % (LossType.Yolo, i)
                modules.add_module(blockName, yolo_layer)
            elif module_def["type"] == LossType.Softmax:
                blockName = "%s_%d" % (LossType.Softmax, i)
                modules.add_module(blockName, EmptyLayer())

            # Register module list and number of output filters
            moduleList.append(modules)
            output_filters.append(filters)
        return moduleList