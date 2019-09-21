import torch.nn as nn
from .baseBlock import *
from .blockName import ActivationType, BlockType

class ActivationLinearLayer(BaseBlock):

    def __init__(self):
        super().__init__(BlockType.ActivationLinearLayer)

    def forward(self, x):
        return x


class ActivationFunction():

    def __init__(self):
        pass

    @classmethod
    def getFunction(self, name):
        if name == ActivationType.Linear:
            return ActivationLinearLayer()
        elif name == ActivationType.ReLU:
            return nn.ReLU()
        elif name == ActivationType.PReLU:
            return nn.PReLU()
        elif name == ActivationType.ReLU6:
            return nn.ReLU6()
        elif name == ActivationType.LeakyReLU:
            return nn.LeakyReLU(0.1)