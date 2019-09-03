import torch
import torch.nn as nn
import torch.nn.functional as F

class ModuleType():

    EmptyLayer = "emptyLayer"
    Upsample = "upsample"
    GlobalAvgPool = "globalavgpool"
    FcLayer = "fcLayer"

    Convolutional = "convolutional"
    ConvBNReLU = "convBNReLU"
    ConvBNLeakyReLU = "convBNLeakyReLU"
    InvertedResidual = "invertedResidual"
    ResidualBasciNeck = "residualBasciNeck"
    ResidualBottleneck = "residualBottleneck"
    SEBlock = "seBlock"
    Maxpool = "maxpool"
    Route = "route"
    Shortcut = "shortcut"
    Softmax = "softmax"
    Yolo = "yolo"

class BaseBlock(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.modelName = name

    def setModelName(self, name):
        self.modelName = name

    def getModelName(self):
        return self.modelName