import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.baseModelName import BaseModelName
from base_model.mobilenetv2 import MobileNetV2
from base_model.shufflenetv2 import ShuffleNetV2

class BaseModelFactory():

    def __init__(self):
        pass

    def getBaseModel(self, baseNetName):
        result = None
        if baseNetName == BaseModelName.ShuffleNetV2:
            result = ShuffleNetV2(net_size=1)
        elif baseNetName == BaseModelName.MobileNetV2:
            result = MobileNetV2(width_mult=1.0)
        return result