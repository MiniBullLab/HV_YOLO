#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import abc


class DataLoader():

    def __init__(self):
        pass

    def all_numpy_to_tensor(self, input_data):
        result = None
        if input_data is None:
            result = None
        elif input_data.ndim == 3:
            result = torch.from_numpy(input_data).unsqueeze(0)
        elif input_data.ndim == 4:
            result = torch.from_numpy(input_data)
        return result

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass
