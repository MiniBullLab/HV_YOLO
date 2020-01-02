#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_block.base_block import *
from base_block.utility_block import InvertedResidual, SEBlock
from drawing.modelNetShow import ModelNetShow


class TestBlock(BaseBlock):
    def __init__(self):
        super().__init__("testBlock")

        # self.conv1 = InvertedResidual(16, 16, 1, 2)
        self.se = SEBlock(16)
        # self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, input):

        x = self.se(input)

        return x

if __name__ == "__main__":

    modelNetShow = ModelNetShow()
    input = torch.randn(1, 16, 32, 32)
    model = TestBlock()
    modelNetShow.setInput(input)
    modelNetShow.setSaveDir("../")
    modelNetShow.showNet(model)
