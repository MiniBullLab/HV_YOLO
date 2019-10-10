import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from collections import OrderedDict
from base_model.baseModel import *
from base_block.blockName import BlockType, LossType
from base_model.baseModelFactory import BaseModelFactory
from .createModelList import *

class MyModel(BaseModel):

    def __init__(self, modelDefine, freezeBn=False):
        super().__init__()
        self.baseModelFactory = BaseModelFactory()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.lossList = []
        self.createModel()
        self.setFreezeBn(freezeBn)

    def createModel(self):
        basicModel = self.creatBaseModel()
        self.add_module(BlockType.BaseNet, basicModel)
        taskBlockDict = self.createTask(basicModel)
        for key, block in taskBlockDict.items():
            self.add_module(key, block)
        self.lossList = self.createLoss(taskBlockDict)

    def creatBaseModel(self):
        result = None
        baseNet = self.modelDefine[0]
        if baseNet["type"] == BlockType.BaseNet:
            baseNetName = baseNet["name"]
            self.modelDefine.pop(0)
            result = self.baseModelFactory.getBaseModel(baseNetName)
        return result

    def createTask(self, basicModel):
        blockDict = OrderedDict()
        outChannels = basicModel.getOutChannelList()
        if basicModel:
            self.createTaskList.createOrderedDict(self.modelDefine, outChannels)
            blockDict = self.createTaskList.getBlockList()
        return blockDict

    def createLoss(self, taskBlockDict):
        lossResult = []
        for key, block in taskBlockDict.items():
            if LossType.Softmax in key:
                lossResult.append(block)
            elif LossType.Yolo in key:
                lossResult.append(block)
        return lossResult

    def setFreezeBn(self, freezeBn):
        for m in self.modules():
            if freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.Convolutional in key:
                x = block(x)
            elif BlockType.ConvActivationBlock in key:
                x = block(x)
            elif BlockType.ConvBNActivationBlock in key:
                x = block(x)
            elif BlockType.Upsample in key:
                x = block(x)
            elif BlockType.Maxpool in key:
                x = block(x)
            elif BlockType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif LossType.Yolo in key:
                output.append(x)
            elif LossType.Softmax in key:
                output.append(x)

            layer_outputs.append(x)
        return output
