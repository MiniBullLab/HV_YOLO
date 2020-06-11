#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class ActivationType():

    Linear = "linear"
    ReLU = "relu"
    PReLU = "prelu"
    ReLU6 = "relu6"
    LeakyReLU = "leaky"
    Sigmoid = "sigmoid"

    Swish = "swish"
    Mish = "mish"

    HardSigmoid = "hard_sigmoid"
    HardSwish = "hard_swish"


class NormalizationType():

    BatchNormalize2d = "bn2d"
    BatchNormalize1d = "bn1d"

    InstanceNorm2d = "in2d"
    InstanceNorm1d = "in1d"


class LayerType():

    EmptyLayer = "emptyLayer"

    MultiplyLayer = "multiply"
    AddLayer = "add"

    NormalizeLayer = "normalize"
    ActivationLayer = "activation"

    RouteLayer = "route"
    ShortRouteLayer = "shortRoute"
    ShortcutLayer = "shortcut"
    MyMaxPool2d = "maxpool"

    Upsample = "upsample"
    PixelShuffle = "pixelShuffle"

    GlobalAvgPool = "globalavgpool"
    FcLayer = "fcLayer"
    Dropout = "dropout"

    FcLinear = "fcLinear"

    Convolutional1d = "convolutional1d"

    Convolutional = "convolutional"

    MeanLayer = "mean"


class BlockType():

    InputData = "inputData"
    BaseNet = "baseNet"

    ConvBNBlock1d = "convBN1d"
    ConvBNActivationBlock1d = "convBNActivationBlock1d"

    ConvBNActivationBlock = "convBNActivationBlock"
    BNActivationConvBlock = "bnActivationConvBlock"
    ActivationConvBNBlock = "activationConvBNBlock"

    ConvActivationBlock = "convActivationBlock"

    FcBNActivationBlock = "fcBNActivationBlock"

    InceptionBlock = "inceptionBlock"

    SeperableConv2dBlock = "seperableConv2dBlock"
    DepthwiseConv2dBlock = "depthwiseConv2dBlock"
    SeparableConv2dBNActivation = "separableConv2dBNActivation"
    ShuffleBlock = "shuffleBlock"

    ResidualBlock = "residualBlock"
    InvertedResidual = "invertedResidual"

    SEBlock = "seBlock"
    SEConvBlock = "seConvBlock"

    Detection2dBlock = "detection2dBlock"
