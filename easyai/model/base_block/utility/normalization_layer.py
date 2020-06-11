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


class L2Norm(nn.Module):
    def __init__(self, input_channel, scale):
        super().__init__()
        self.input_channel = input_channel
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.input_channel))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


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
