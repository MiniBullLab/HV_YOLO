#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from model.model_factory import ModelFactory
from torch_utility.torchOptimizer import TorchOptimizer


def main():
    optimizerM = {0: {'optimizer': 'SGD',
                     'lr': 1e-2,
                     'momentum': 0.9,
                     'weight_decay': 5e-4},
                 2: {'optimizer': 'Adam',
                     'momentum': 0.9,
                     'weight_decay': 5e-4},
                 4: {'optimizer': 'SGD',
                     'lr': 1e-3,
                     'momentum': 0.9,
                     'weight_decay': 5e-4}
                }

    model_factory = ModelFactory
    model = model_factory.getModelFromName("MobileV2FCN")

    for epoch in range(0, 5):
        print("epoch {}...............".format(epoch))
        optimizerMethod = TorchOptimizer(model, epoch, optimizerM)
        optimizerMethod.adjust_optimizer()


if __name__ == "__main__":
    main()