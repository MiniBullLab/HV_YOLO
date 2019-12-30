#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

class ActivationType():

    Linear = "linear"
    ReLU = "relu"
    PReLU = "prelu"
    ReLU6 = "relu6"
    LeakyReLU = "leaky"

class BatchNormType():

    BatchNormalize = "bn2d"

class BlockType():

    InputData = "inputData"
    BaseNet = "baseNet"

    LinearActivation = "linearActivation"
    EmptyLayer = "emptyLayer"
    RouteLayer = "route"
    ShortcutLayer = "shortcut"
    MyMaxPool2d = "maxpool"
    Upsample = "upsample"
    GlobalAvgPool = "globalavgpool"
    FcLayer = "fcLayer"

    ConvBNActivationBlock = "convBNActivationBlock"
    ConvActivationBlock = "convActivationBlock"
    Convolutional = "convolutional"

    InvertedResidual = "invertedResidual"
    SEBlock = "seBlock"
