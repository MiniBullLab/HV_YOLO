import torch
import random
import numpy as np

class TorchDeviceProcess():

    device = "cuda:0"
    cuda = None

    def __init__(self):
        self.cude = None
        self.count = 1

    @classmethod
    def hasCUDA(cls):
        cls.cuda = torch.cuda.is_available()
        if cls.cuda:
            cls.device = "cuda:0"
            return True
        else:
            cls.device = "cpu"
            return False

    def getCUDACount(self):
        self.count = torch.cuda.device_count()
        return self.count

    def initTorch(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if self.hasCUDA():
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            print("Using device: \"{}\"".format(self.device))





