import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from model.modelParse import ModelParse
from .baseModelName import BaseModelName
from .mobilenetv2 import MobileNetV2
from .shufflenetv2 import ShuffleNetV2
from .my_base_model import MyBaseModel

class BaseModelFactory():

    def __init__(self):
        cfgReader = ModelParse()

    def getBaseModel(self, baseNetName):
        input_name = baseNetName.strip()
        if input_name.endswith("cfg"):
            result = self.get_base_model_from_cfg(input_name)
        else:
            result = self.get_base_model_from_name(input_name)
        return result

    def get_base_model_from_cfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.readCfgFile(cfg_path)
        model = MyBaseModel(model_define)
        model.setModelName(file_name)
        return model

    def get_base_model_from_name(self, net_name):
        result = None
        if net_name == BaseModelName.ShuffleNetV2:
            result = ShuffleNetV2()
        elif net_name == BaseModelName.MobileNetV2:
            result = MobileNetV2()
        return result
