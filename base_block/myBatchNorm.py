import torch.nn as nn
from .blockName import BatchNormType, BlockType

class MyBatchNormalize():

    def __init__(self):
        pass

    @classmethod
    def getFunction(cls, name, inputChannel):
        if name == BatchNormType.BatchNormalize:
            return nn.BatchNorm2d(inputChannel)