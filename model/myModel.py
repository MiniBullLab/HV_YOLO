import torch
import torch.nn as nn
from collections import defaultdict
from base_model.baseLayer import EmptyLayer, Upsample
from base_model.shufflenetv2 import ShuffleNetV2
from loss import *

class MyModel(nn.Module):

    def __init__(self, modelDefine, freezeBn=False):
        super().__init__()
        self.modelDefine = modelDefine
        self.freezeBn = freezeBn
        self.basicModel = self.creatBaseModel()
        self.taskModules = self.createTask()
        self.lossList = self.createLoss()
        self.setFreezeBn()

    def creatBaseModel(self):
        result = None
        baseNet = self.modelDefine[0]
        if baseNet["type"] == "baseNet":
            baseNetName = baseNet["name"]
            self.modelDefine.pop(0)
            if baseNetName == "ShuffleNetV2":
                result = ShuffleNetV2(net_size=1)
        return result

    def createTask(self):
        moduleList = nn.ModuleList()
        if self.basicModel:
            outChannels = self.basicModel.out_channels
            moduleList = self.getModuleList(outChannels)
        return moduleList

    def createLoss(self):
        # yoloLoss
        yoloLoss = []
        for m in self.taskModules:
            for layer in m:
                if isinstance(layer, YoloLoss):
                    yoloLoss.append(layer)
        return yoloLoss

    def getModuleList(self, inputChannels):
        output_filters = [inputChannels[-1]]
        moduleList = nn.ModuleList()
        for i, module_def in enumerate(self.modelDefine):
            modules = nn.Sequential()

            if module_def['type'] == 'convolutional':
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

            elif module_def['type'] == 'maxpool':
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                modules.add_module('maxpool_%d' % i, maxpool)

            elif module_def['type'] == 'upsample':
                # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
                upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
                modules.add_module('upsample_%d' % i, upsample)

            elif module_def['type'] == 'route':
                layers = [int(x) for x in module_def['layers'].split(',')]
                filters = sum([inputChannels[i] if i >= 0 else output_filters[i] for i in layers])
                # filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                modules.add_module('route_%d' % i, EmptyLayer())

            elif module_def['type'] == 'shortcut':
                filters = output_filters[int(module_def['from'])]
                modules.add_module('shortcut_%d' % i, EmptyLayer())

            elif module_def['type'] == 'yolo':
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                num_classes = int(module_def['classes'])
                # Define detection layer
                yolo_layer = YoloLoss(num_classes, anchors=anchors, anchors_mask=anchor_idxs, smoothLabel=False,
                                      focalLoss=False)
                modules.add_module('yolo_%d' % i, yolo_layer)

            # Register module list and number of output filters
            moduleList.append(modules)
            output_filters.append(filters)
        return moduleList

    def setFreezeBn(self):
        for m in self.modules():
            if self.freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        self.losses = defaultdict(float)
        layer_outputs = []
        output = []

        blocks = self.basicModel(x)
        x = blocks[-1]

        for i, (module_def, module) in enumerate(zip(self.modelDefine, self.taskModules)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] if i < 0 else blocks[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                output.append(x)

            layer_outputs.append(x)

        return output