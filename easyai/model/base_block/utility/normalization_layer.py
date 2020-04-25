#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import torch.nn as nn
from easyai.base_name.block_name import NormalizationType


class NormalizationFunction():

    def __init__(self):
        pass

    @classmethod
    def get_function(cls, name, input_channel):
        if name == NormalizationType.BatchNormalize2d:
            return nn.BatchNorm2d(input_channel)
        elif name == NormalizationType.BatchNormalize1d:
            return nn.BatchNorm1d(input_channel)
        elif name == NormalizationType.InstanceNorm2d:
            return nn.InstanceNorm2d(input_channel)
        elif name == NormalizationType.BatchNormalize1d:
            return nn.InstanceNorm1d(input_channel)
