#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
from base_name.base_model_name import BaseModelName
from base_model.mobilenetv2 import MobileNetV2
from base_model.shufflenetv2 import ShuffleNetV2
from base_model.my_base_model import MyBaseModel
from model.modelParse import ModelParse


class BaseModelFactory():

    def __init__(self):
        self.cfgReader = ModelParse()

    def get_base_model(self, baseNetName):
        input_name = baseNetName.strip()
        if input_name.endswith("cfg"):
            result = self.get_base_model_from_cfg(input_name)
        else:
            result = self.get_base_model_from_name(input_name)
        return result

    def get_base_model_from_cfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.cfgReader.readCfgFile(cfg_path)
        model = MyBaseModel(model_define)
        model.set_name(file_name)
        return model

    def get_base_model_from_name(self, net_name):
        result = None
        if net_name == BaseModelName.ShuffleNetV2:
            result = ShuffleNetV2()
        elif net_name == BaseModelName.MobileNetV2:
            result = MobileNetV2()
        return result
