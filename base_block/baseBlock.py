import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BaseBlock(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.modelName = name

    def setModelName(self, name):
        self.modelName = name

    def getModelName(self):
        return self.modelName