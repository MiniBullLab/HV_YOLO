import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.modelName = "None"

    def setModelName(self, name):
        self.modelName = name

    def getModelName(self):
        return self.modelName