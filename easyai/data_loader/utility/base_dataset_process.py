#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
import torch
from easyai.torch_utility.torch_vision.torchvision_process import TorchVisionProcess


class BaseDataSetProcess():

    def __init__(self):
        self.torchvision_process = TorchVisionProcess()

    def numpy_to_torch(self, data, flag=0):
        result = None
        if flag == 0:
            result = torch.from_numpy(data)
        elif flag == 1:  # torchvision to tensor
            # data(numpy or PIL.Image) convert to 0~1.0 and tensor(C,H, W)
            transform = self.torchvision_process.torch_normalize(flag=1)
            result = transform(data)
        return result
