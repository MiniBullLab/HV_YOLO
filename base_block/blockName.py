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

    BaseNet = "baseNet"

    ActivationLinearLayer = "activationLinearLayer"
    EmptyLayer = "emptyLayer"
    RouteLayer = "route"
    ShortcutLayer = "shortcut"
    Maxpool = "maxpool"
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

class LossType():

    Softmax = "softmax"
    Yolo = "yolo"