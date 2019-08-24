import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

class ModuleType():

    Convolutional = "convolutional"
    Maxpool = "maxpool"
    GlobalAvgPool = "globalavgpool"
    Upsample = "upsample"
    Route = "route"
    Shortcut = "shortcut"
    Softmax = "softmax"
    Yolo = "yolo"

class ModuleName():

    MobileV2FCN = "MobileV2FCN"

class BaseModuleName():

    ShuffleNetV2 = "ShuffleNetV2"
    MobileNetV2 = "MobileNetV2"

