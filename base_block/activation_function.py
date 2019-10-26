#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from base_block.base_block import *
from base_block.block_name import ActivationType, BlockType

"""
"""
class LinearActivation(BaseBlock):

    def __init__(self):
        super().__init__(BlockType.LinearActivation)

    def forward(self, x):
        return x


class ActivationFunction():

    def __init__(self):
        pass

    @classmethod
    def get_function(cls, name):
        if name == ActivationType.Linear:
            return LinearActivation()
        elif name == ActivationType.ReLU:
            return nn.ReLU()
        elif name == ActivationType.PReLU:
            return nn.PReLU()
        elif name == ActivationType.ReLU6:
            return nn.ReLU6()
        elif name == ActivationType.LeakyReLU:
            return nn.LeakyReLU(0.1)
