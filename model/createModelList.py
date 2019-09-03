import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch.nn as nn
from base_block.baseBlock import ModuleType
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

            if module_def['type'] == ModuleType.Convolutional:
                bn = int(module_def['batch_normalize'])
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(module_def['stride']),
                                                            padding=pad,
                                                            bias=not bn))
                if bn:
                    modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))
                elif module_def['activation'] == 'relu':
                    modules.add_module('leaky_%d' % i, nn.ReLU())

            elif module_def['type'] == ModuleType.Maxpool:
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                modules.add_module('Maxpool_%d' % i, maxpool)

            ###############################################
            elif module_def['type'] == ModuleType.GlobalAvgPool:
                globalAvgPool = GlobalAvgPool2d()
                modules.add_module('GlobalAvgPool' % i, globalAvgPool)

            elif module_def['type'] == ModuleType.Upsample:
                # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
                #upsample = Upsample(scale_factor=int(module_def['stride']), mode='bilinear')
                upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
                modules.add_module('upsample_%d' % i, upsample)

            elif module_def['type'] == ModuleType.Route:
                layers = [int(x) for x in module_def['layers'].split(',')]
                filters = sum([inputChannels[i] if i >= 0 else output_filters[i] for i in layers])
                # filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                modules.add_module('route_%d' % i, EmptyLayer())

            elif module_def['type'] == ModuleType.Shortcut:
                filters = output_filters[int(module_def['from'])]
                modules.add_module('shortcut_%d' % i, EmptyLayer())

            elif module_def['type'] == ModuleType.Yolo:
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                num_classes = int(module_def['classes'])
                # Define detection layer
                yolo_layer = YoloLoss(num_classes, anchors=anchors, anchors_mask=anchor_idxs, smoothLabel=False,
                                      focalLoss=False)
                modules.add_module('yolo_%d' % i, yolo_layer)
            elif module_def["type"] == ModuleType.Softmax:
                modules.add_module('softmax_%d' % i, EmptyLayer())

            # Register module list and number of output filters
            moduleList.append(modules)
            output_filters.append(filters)
        return moduleList