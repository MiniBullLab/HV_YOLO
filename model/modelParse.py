import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from .modelName import  ModelName
from .myModel import MyModel
#from .mobileV2FCN import MobileV2FCN

class ModelParse():

    def __init__(self):
        pass

    def parse(self, cfgPath, freezeBn=False):
        path, fileNameAndPost = os.path.split(cfgPath)
        fileName, post = os.path.splitext(fileNameAndPost)
        modelDefine = self.readCfgFile(cfgPath)
        model = MyModel(modelDefine, freezeBn)
        model.setModelName(fileName)
        return model

    def readCfgFile(self, cfgPath):
        modelDefine = []
        file = open(cfgPath, 'r')
        lines = file.read().split('\n')
        lines = [x.strip() for x in lines if x.strip() and not x.startswith('#')]
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                modelDefine.append({})
                modelDefine[-1]['type'] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                value = value.strip()
                modelDefine[-1][key.strip()] = value.strip()
        #print(modelDefine)
        return modelDefine

    def getModel(self, modelName):
        model = None
        if modelName == ModelName.MobileV2FCN:
            model = MobileV2FCN()
        return model