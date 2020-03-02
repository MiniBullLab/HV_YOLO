#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.model.base_block.utility.base_block import *


class LinearActivation(BaseBlock):

    def __init__(self):
        super().__init__(ActivationType.Linear)

    def forward(self, x):
        return x


class MishActivation(BaseBlock):

    def __init__(self):
        super().__init__(ActivationType.Mish)

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class ActivationFunction():

    def __init__(self):
        pass

    @classmethod
    def get_function(cls, name, inplace=True):
        if name == ActivationType.Linear:
            return LinearActivation()
        elif name == ActivationType.ReLU:
            return nn.ReLU(inplace=inplace)
        elif name == ActivationType.PReLU:
            return nn.PReLU()
        elif name == ActivationType.ReLU6:
            return nn.ReLU6(inplace=inplace)
        elif name == ActivationType.LeakyReLU:
            return nn.LeakyReLU(0.1, inplace=inplace)
        elif name == ActivationType.Sigmoid:
            return nn.Sigmoid()
