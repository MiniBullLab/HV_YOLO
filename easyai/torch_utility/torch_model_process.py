import os.path
import torch
from torch import nn
from collections import OrderedDict
from easyai.torch_utility.torch_device_process import TorchDeviceProcess
from easyai.model.utility.model_factory import ModelFactory
from easyai.model.utility.mode_weight_init import ModelWeightInit


class TorchModelProcess():

    def __init__(self):
        self.torchDeviceProcess = TorchDeviceProcess()
        self.modelFactory = ModelFactory()
        self.modelWeightInit = ModelWeightInit()
        self.torchDeviceProcess.initTorch()
        self.best_value = 0
        self.is_multi_gpu = False

    def initModel(self, cfgPath, gpuId, is_multi_gpu=False):
        self.is_multi_gpu = is_multi_gpu
        self.torchDeviceProcess.setGpuId(gpuId)
        model = self.modelFactory.get_model(cfgPath)
        self.modelWeightInit.initWeight(model)
        return model

    def loadPretainModel(self, weightPath, model):
        if os.path.exists(weightPath):
            print("Loading pretainModel from {}".format(weightPath))
            model.load_state_dict(torch.load(weightPath), strict=True)
        else:
            print("Loading %s fail" % weightPath)

    def loadLatestModelWeight(self, weightPath, model):
        count = self.torchDeviceProcess.getCUDACount()
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

    def saveLatestModel(self, latestWeightsFile, model, optimizer, epoch=0, best_value=0):
        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_value': best_value,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latestWeightsFile)

    def getLatestModelValue(self, checkpoint):
        start_epoch = 0
        value = -1
        if checkpoint:
            if checkpoint.get('epoch') is not None:
                start_epoch = checkpoint['epoch'] + 1
            if checkpoint.get('best_value') is not None:
                value = checkpoint['best_value']
        self.setModelBestValue(value)
        return start_epoch, value

    def setModelBestValue(self, value):
        self.best_value = value

    def saveBestModel(self, value, latest_weights_file, best_weights_file):
        if value >= self.best_value:
            self.best_value = value
            os.system('cp {} {}'.format(
                latest_weights_file,
                best_weights_file,
            ))
        return self.best_value

    def modelTrainInit(self, model):
        count = self.torchDeviceProcess.getCUDACount()
        if count > 1 and self.is_multi_gpu:
            print('Using ', count, ' GPUs')
            model = nn.DataParallel(model)
        model = model.to(self.torchDeviceProcess.device)
        model.train()

    def modelTestInit(self, model):
        model = model.to(self.torchDeviceProcess.device)
        model.eval()

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

    def getDevice(self):
        return self.torchDeviceProcess.device
