#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easy_ai.base_name import ActivationType


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
    def get_function(cls, name):
        if name == ActivationType.Linear:
            return LinearActivation()
        elif name == ActivationType.ReLU:
            return nn.ReLU(inplace=True)
        elif name == ActivationType.PReLU:
            return nn.PReLU()
        elif name == ActivationType.ReLU6:
            return nn.ReLU6(inplace=True)
        elif name == ActivationType.LeakyReLU:
            return nn.LeakyReLU(0.1, inplace=True)
        elif name == ActivationType.Sigmoid:
            return nn.Sigmoid()
