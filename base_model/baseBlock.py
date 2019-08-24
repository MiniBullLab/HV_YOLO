import torch
import torch.nn as nn

class BaseBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.modelName = "none"

    def setModelName(self, name):
        self.modelName = name

    def getModelName(self):
        return self.modelName