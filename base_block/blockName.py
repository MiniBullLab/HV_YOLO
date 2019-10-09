import torch.nn as nn

class ActivationType():

    Linear = "linear"
    ReLU = "relu"
    PReLU = "prelu"
    ReLU6 = "relu6"
    LeakyReLU = "leaky"

class BatchNormType():

    BatchNormalize = "bn2d"

class BlockType():

    ActivationLinearLayer = "activationLinearLayer"
    EmptyLayer = "emptyLayer"
    Upsample = "upsample"
    GlobalAvgPool = "globalavgpool"
    FcLayer = "fcLayer"

    ConvBNActivationBlock = "convBNActivationBlock"
    ConvActivationBlock = "convActivationBlock"
    Convolutional = "convolutional"

    InvertedResidual = "invertedResidual"
    ResidualBasciNeck = "residualBasciNeck"
    ResidualBottleneck = "residualBottleneck"
    SEBlock = "seBlock"

    Maxpool = "maxpool"
    Route = "route"
    Shortcut = "shortcut"

class LossType():

    Softmax = "softmax"
    Yolo = "yolo"