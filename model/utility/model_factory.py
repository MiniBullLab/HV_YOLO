#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from base_name.model_name import ModelName
from model.utility.modelParse import ModelParse
from model.utility.myModel import MyModel
from model.sr.MSRResNet import MSRResNet
from model.sr.MySRModel import MySRModel
from model.seg.mobileV2FCN import MobileV2FCN
from model.det3d.complex_yolo import ComplexYOLO


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
        model = MyModel(model_define)
        model.set_name(file_name)
        return model

    def get_model_from_name(self, modelName):
        model = None
        if modelName == ModelName.MSRResNet:
            model = MSRResNet(in_nc=1, upscale_factor=3)
        elif modelName == ModelName.MySRModel:
            model = MySRModel(upscale_factor=3)
        elif modelName == ModelName.MobileV2FCN:
            model = MobileV2FCN()
        elif modelName == ModelName.ComplexYOLO:
            model = ComplexYOLO()
        return model
