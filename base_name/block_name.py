#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class ActivationType():

    Mish = "mish"
    Linear = "linear"
    ReLU = "relu"
    PReLU = "prelu"
    ReLU6 = "relu6"
    LeakyReLU = "leaky"
    Sigmoid = "sigmoid"


class BatchNormType():

    BatchNormalize = "bn2d"


class BlockType():

    InputData = "inputData"
    BaseNet = "baseNet"

    LinearActivation = "linearActivation"

    EmptyLayer = "emptyLayer"

    MultiplyLayer = "multiply"
    AddLayer = "add"

    RouteLayer = "route"
    ShortRouteLayer = "shortRoute"
    ShortcutLayer = "shortcut"
    MyMaxPool2d = "maxpool"
    Upsample = "upsample"
    GlobalAvgPool = "globalavgpool"
    FcLayer = "fcLayer"
    Dropout = "dropout"

    ConvBNActivationBlock = "convBNActivationBlock"
    ConvActivationBlock = "convActivationBlock"
    Convolutional = "convolutional"
    BNActivationConvBlock = "bnActivationConvBlock"
    BNActivationBlock = "bnActivationBlock"

    InvertedResidual = "invertedResidual"
    SEBlock = "seBlock"
