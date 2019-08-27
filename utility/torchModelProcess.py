import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from torch import nn
from collections import OrderedDict
from .torchDeviceProcess import TorchDeviceProcess
from model.modelParse import ModelParse

class TorchModelProcess():

    def __init__(self):
        self.torchDeviceProcess = TorchDeviceProcess()
        self.modelParse = ModelParse()

    def initModel(self, cfgPath):
        self.torchDeviceProcess.initTorch();
        model = self.modelParse.parse(cfgPath, freezeBn=False)
        return model

    def loadPretainModel(self, weightPath, model):
        if os.path.exists(weightPath):
            print("Loading pretainModel from {}".format(weightPath))
            model.load_state_dict(torch.load(weightPath), strict=True)
        else:
            print("Loading %s fail" % weightPath)

    def loadLatestModelWeight(self, weightPath, model):
        count = self.torchDeviceProcess.getCUDACount();
        checkpoint = None
        if os.path.exists(weightPath):
            if count > 1:
                checkpoint = torch.load(weightPath, map_location='cpu')
                state = self.convert_state_dict(checkpoint['model'])
                model.load_state_dict(state)
            else:
                checkpoint = torch.load(weightPath, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
        else:
            print("Loading %s fail" % weightPath)
        return checkpoint

    def saveLatestModel(self, latestWeightsFile, model, optimizer, epoch=0, best_mAP=0):
        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_mAP': best_mAP,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latestWeightsFile)

    def modelTrainInit(self, model):
        count = self.torchDeviceProcess.getCUDACount()
        if count > 1:
            print('Using ', count, ' GPUs')
            model = nn.DataParallel(model)
        model.to(TorchDeviceProcess.device).train()

    def modelTestInit(self, model):
        model.to(TorchDeviceProcess.device).eval()

    def convert_state_dict(self, state_dict):
        """Converts a state dict saved from a dataParallel module to normal
           module state_dict inplace
           :param state_dict is the loaded DataParallel model_state

        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict