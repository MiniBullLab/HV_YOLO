import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from .myModel import MyModel

class ModelParse():

    def __init__(self):
        pass

    def parse(self, cfgPath, freezeBn=False):
        modelDefine = self.readCfgFile(cfgPath)
        model = MyModel(modelDefine, freezeBn)
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
                if modelDefine[-1]['type'] == 'convolutional':
                    modelDefine[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                modelDefine[-1][key.strip()] = value.strip()
        print(modelDefine)
        return modelDefine