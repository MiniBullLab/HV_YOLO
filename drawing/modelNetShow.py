import os
import os.path
import netron
import torch
from torch import onnx

class ModelNetShow():

    def __init__(self, channel=3, width=224, height=224):
        self.input = torch.ones(1, channel, width, height)

    def showNet(self, model):
        #print(model.state_dict())
        self.__torch2onnx(model, self.input)

    def __torch2onnx(self, model, input):
        saveModelPath = os.path.join("./onnx", "%s.onnx" % model.getModelName())
        #print(saveModelPath)
        onnx.export(model, input, saveModelPath)
        netron.start(saveModelPath, port=9999, host='localhost')