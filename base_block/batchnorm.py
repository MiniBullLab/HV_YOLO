#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import torch.nn as nn
from base_name.block_name import BatchNormType


class BatchNormalizeFunction():

    def __init__(self):
        pass

    @classmethod
    def get_function(cls, name, input_channel):
        if name == BatchNormType.BatchNormalize:
            return nn.BatchNorm2d(input_channel)
