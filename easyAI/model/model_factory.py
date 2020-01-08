#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyAI.base_name.model_name import ModelName
from easyAI.model.modelParse import ModelParse
from easyAI.model.myModel import MyModel
from easyAI.model.mobileV2FCN import MobileV2FCN


class ModelFactory():

    def __init__(self):
        self.modelParse = ModelParse()

    def get_model(self, input_name):
        input_name = input_name.strip()
        if input_name.endswith("cfg"):
            result = self.get_model_from_cfg(input_name)
        else:
            result = self.get_model_from_name(input_name)
        return result

    def get_model_from_cfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.modelParse.readCfgFile(cfg_path)
        model = MyModel(model_define, path)
        model.set_name(file_name)
        return model

    def get_model_from_name(self, modelName):
        model = None
        if modelName == ModelName.MobileV2FCN:
            model = MobileV2FCN()
        else:
            print("%s error" % modelName)
        return model
