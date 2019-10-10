import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from .baseModel import *
from model.createModelList import *

class MyBaseModel(BaseModel):

    def __init__(self, modelDefine, freezeBn=False):
        super().__init__()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.outChannelList = []
        self.createModel()
        self.setFreezeBn(freezeBn)

    def createModel(self):
        outChannels = []
        self.createTaskList.createOrderedDict(self.modelDefine, outChannels)
        blockDict = self.createTaskList.getBlockList()
        self.outChannelList = self.getOutChannelList()
        for key, block in blockDict.items():
            name = "base_%s" % key
            self.add_module(name, block)

    def getOutChannelList(self):
        return self.outChannelList

    def setFreezeBn(self, freezeBn):
        for m in self.modules():
            if freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.Convolutional in key:
                x = block(x)
            elif BlockType.ConvActivationBlock in key:
                x = block(x)
            elif BlockType.ConvBNActivationBlock in key:
                x = block(x)
            elif BlockType.Upsample in key:
                x = block(x)
            elif BlockType.Maxpool in key:
                x = block(x)
            elif BlockType.Route in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.Shortcut in key:
                x = block(layer_outputs)
            elif LossType.Yolo in key:
                output.append(x)
            elif LossType.Softmax in key:
                output.append(x)

            layer_outputs.append(x)
        return layer_outputs
