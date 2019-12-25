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

    BatchNormalize2d = "bn2d"
    BatchNormalize1d = "bn1d"


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

    FcLinear = "fcLinear"

    Convolutional1d = "convolutional1d"
    ConvBNBlock1d = "convBN1d"
    ConvBNActivationBlock1d = "convBNActivationBlock1d"

    Convolutional = "convolutional"
    ConvBNActivationBlock = "convBNActivationBlock"
    ConvActivationBlock = "convActivationBlock"
    BNActivationConvBlock = "bnActivationConvBlock"
    BNActivationBlock = "bnActivationBlock"

    FcBNActivationBlock = "fcBNActivationBlock"

    InvertedResidual = "invertedResidual"
    SEBlock = "seBlock"
