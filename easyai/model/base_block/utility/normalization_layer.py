#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import torch.nn as nn
from easyai.base_name.block_name import NormalizationType


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called "weight" and "bias".
    The two buffers are computed from the original four parameters of BN:
    mean, variance, scale (gamma), offset (beta).
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - mean) / std * scale + offset`, but will be slightly cheaper.
    The pre-trained backbone models from Caffe2 are already in such a frozen format.
    """
    def __init__(self, input_channel):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(input_channel))
        self.register_buffer("bias", torch.zeros(input_channel))

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class NormalizationFunction():

    def __init__(self):
        pass

    @classmethod
    def get_function(cls, name, input_channel, momentum=0.1):
        if name == NormalizationType.BatchNormalize2d:
            return nn.BatchNorm2d(input_channel, momentum=momentum)
        elif name == NormalizationType.BatchNormalize1d:
            return nn.BatchNorm1d(input_channel, momentum=momentum)
        elif name == NormalizationType.InstanceNorm2d:
            return nn.InstanceNorm2d(input_channel, momentum=0.1)
        elif name == NormalizationType.BatchNormalize1d:
            return nn.InstanceNorm1d(input_channel, momentum=0.1)
