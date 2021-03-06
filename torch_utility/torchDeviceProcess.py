import torch
import random
import numpy as np

class TorchDeviceProcess():

    first_run = True
    cuda = None

    def __init__(self):
        self.device = "cpu"
        self.setGpuId(0)

    @classmethod
    def hasCUDA(cls):
        cls.cuda = torch.cuda.is_available()
        if cls.cuda:
            return True
        else:
            return False

    def setGpuId(self, id):
        if TorchDeviceProcess.cuda:
            count = self.getCUDACount()
            if id >= 0 and id < count:
                self.device = "cuda:%d" % id
        else:
            self.device = "cpu"

    def getCUDACount(self):
        count = torch.cuda.device_count()
        return count

    def initTorch(self, gpuId):
        if TorchDeviceProcess.first_run:
            torch.cuda.empty_cache()
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if TorchDeviceProcess.hasCUDA():
                self.setGpuId(gpuId)
                torch.cuda.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.benchmark = True
                print("Using device: \"{}\"".format(self.device))
            TorchDeviceProcess.first_run = False





