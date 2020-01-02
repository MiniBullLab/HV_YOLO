import os
import os.path
import netron
import torch
from torch import onnx

class ModelNetShow():

    def __init__(self, channel=3, width=224, height=224):
        self.input = torch.ones(1, channel, width, height)
        self.saveDir = "./onnx"

    def setInput(self, inputTorch):
        self.input = inputTorch

    def setSaveDir(self, saveDir):
        self.saveDir = saveDir

    def showNet(self, model):
        #print(model.state_dict())
        self.__torch2onnx(model, self.input)

    def __torch2onnx(self, model, input):
        saveModelPath = os.path.join(self.saveDir, "%s.onnx" % model.get_name())
        #print(saveModelPath)
        onnx.export(model, input, saveModelPath)
        netron.start(saveModelPath, port=9999, host='localhost')