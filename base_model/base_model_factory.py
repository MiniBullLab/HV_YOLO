#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
from base_name.base_model_name import BaseModelName
from base_model.mobilenetv2 import mobilenetv2_1_0
from base_model.shufflenetv2 import shufflenetv2_1_0
from base_model.resnet import resnet18, resnet34
from base_model.resnet import resnet50, resnet101, resnet152
from base_model.darknet import darknet21, darknet53
from base_model.darknet import darknet21_dilated8, darknet21_dilated16
from base_model.darknet import darknet53_dilated8, darknet53_dilated16
from base_model.googlenet import GoogleNet
from base_model.vgg import vgg13, vgg16, vgg19
from base_model.squeezenet import SqueezeNet, DilatedSqueezeNet
from base_model.densenet import densenet121, densenet201, densenet169, densenet161
from base_model.densenet import densenet121_dilated8, densenet121_dilated16
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
        if net_name == BaseModelName.ShuffleNetV2_1_0:
            result = shufflenetv2_1_0()
        elif net_name == BaseModelName.MobileNetV2_1_0:
            result = mobilenetv2_1_0()
        elif net_name == BaseModelName.ResNet18:
            result = resnet18()
        elif net_name == BaseModelName.ResNet34:
            result = resnet34()
        elif net_name == BaseModelName.ResNet50:
            result = resnet50()
        elif net_name == BaseModelName.ResNet101:
            result = resnet101()
        elif net_name == BaseModelName.ResNet152:
            result = resnet152()
        elif net_name == BaseModelName.Darknet53:
            result = darknet53()
        elif net_name == BaseModelName.Darknet21:
            result = darknet21()
        elif net_name == BaseModelName.Darknet21_Dilated8:
            result = darknet21_dilated8()
        elif net_name == BaseModelName.Darknet21_Dilated16:
            result = darknet21_dilated16()
        elif net_name == BaseModelName.Darknet53_Dilated8:
            result = darknet53_dilated8()
        elif net_name == BaseModelName.Darknet53_Dilated16:
            result = darknet53_dilated16()
        elif net_name == BaseModelName.GoogleNet:
            result = GoogleNet()
        elif net_name == BaseModelName.Vgg13:
            result = vgg13()
        elif net_name == BaseModelName.Vgg16:
            result = vgg16()
        elif net_name == BaseModelName.Vgg19:
            result = vgg19()
        elif net_name == BaseModelName.SqueezeNet:
            result = SqueezeNet()
        elif net_name == BaseModelName.DilatedSqueezeNet:
            result = DilatedSqueezeNet()
        elif net_name == BaseModelName.Densenet121:
            result = densenet121()
        elif net_name == BaseModelName.Densenet121_Dilated8:
            result = densenet121_dilated8()
        elif net_name == BaseModelName.Densenet121_Dilated16:
            result = densenet121_dilated16()
        elif net_name == BaseModelName.Densenet169:
            result = densenet169()
        elif net_name == BaseModelName.Densenet201:
            result = densenet201()
        elif net_name == BaseModelName.Densenet161:
            result = densenet161()
        else:
            print("base model:%s error" % net_name)
        return result
