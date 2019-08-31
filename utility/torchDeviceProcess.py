import torch
import random
import numpy as np

class TorchDeviceProcess():

    def __init__(self):
        self.cuda = None
        self.device = "cpu"
        self.hasCUDA()
        self.setGpuId(0)

    def hasCUDA(self):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            return True
        else:
            return False

    def setGpuId(self, id):
        if self.cuda:
            count = self.getCUDACount()
            if id >= 0 and id < count:
                self.device = "cuda:%d" % id
        else:
            self.device = "cpu"

    def getCUDACount(self):
        count = torch.cuda.device_count()
        return count

    def initTorch(self, gpuId):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if self.hasCUDA():
            self.setGpuId(gpuId)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            print("Using device: \"{}\"".format(self.device))





