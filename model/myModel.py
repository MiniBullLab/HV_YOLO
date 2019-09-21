import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.baseModel import *
from collections import defaultdict
from .createModelList import *
from base_block.blockName import BlockType, LossType
from .baseModelFactory import BaseModelFactory

class MyModel(BaseModel):

    def __init__(self, modelDefine, freezeBn=False):
        super().__init__()
        self.baseModelFactory = BaseModelFactory()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.freezeBn = freezeBn
        self.basicModel = self.creatBaseModel()
        self.taskModules = self.createTask()
        self.lossList = self.createLoss()
        self.setFreezeBn(freezeBn)

    def creatBaseModel(self):
        result = None
        baseNet = self.modelDefine[0]
        if baseNet["type"] == "baseNet":
            baseNetName = baseNet["name"]
            self.modelDefine.pop(0)
            result = self.baseModelFactory.getBaseModel(baseNetName)
        return result

    def createTask(self):
        moduleList = nn.ModuleList()
        if self.basicModel:
            outChannels = self.basicModel.out_channels
            moduleList = self.createTaskList.getModuleList(outChannels, self.modelDefine)
        return moduleList

    def createLoss(self):
        # yoloLoss
        lossResult = []
        for m in self.taskModules:
            for layer in m:
                if isinstance(layer, YoloLoss):
                    lossResult.append(layer)
        return lossResult

    def setFreezeBn(self, freezeBn):
        for m in self.modules():
            if freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        self.losses = defaultdict(float)
        layer_outputs = []
        output = []

        blocks = self.basicModel(x)
        x = blocks[-1]

        for i, (module_def, module) in enumerate(zip(self.modelDefine, self.taskModules)):
            if module_def['type'] in [BlockType.Convolutional, BlockType.ConvActivationBlock,\
                                      BlockType.ConvBNActivationBlock]:
                x = module(x)
            elif module_def['type'] in [BlockType.Upsample, BlockType.Maxpool]:
                x = module(x)
            elif module_def['type'] == BlockType.Route:
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] if i < 0 else blocks[i] for i in layer_i], 1)
            elif module_def['type'] == BlockType.Shortcut:
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == LossType.Yolo:
                output.append(x)
            elif module_def['type'] == LossType.Softmax:
                output.append(x)

            layer_outputs.append(x)

        return output