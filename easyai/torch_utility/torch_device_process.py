import os
import torch
import random
import numpy as np


class TorchDeviceProcess():

    first_run = True
    cuda = None

    def __init__(self):
        self.device = torch.device("cpu")

    @classmethod
    def hasCUDA(cls):
        cls.cuda = torch.cuda.is_available()
        if cls.cuda:
            return True
        else:
            return False

    def setGpuId(self, id=0):
        if TorchDeviceProcess.cuda:
            count = self.getCUDACount()
            if id >= 0 and id < count:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
            else:
                print("GPU %d error" % id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using device: \"{}\"".format(self.device))

    def getCUDACount(self):
        count = -1
        if TorchDeviceProcess.cuda:
            count = torch.cuda.device_count()
        return count

    def initTorch(self):
        if TorchDeviceProcess.first_run:
            torch.cuda.empty_cache()
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if TorchDeviceProcess.hasCUDA():
                torch.cuda.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.benchmark = True
            TorchDeviceProcess.first_run = False