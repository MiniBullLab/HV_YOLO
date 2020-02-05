#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.cls.mobilenetv2 import mobilenetv2_1_0
from easyai.model.backbone.cls.shufflenetv2 import shufflenetv2_1_0
from easyai.model.backbone.cls.resnet import resnet18, resnet34
from easyai.model.backbone.cls.resnet import resnet50, resnet101, resnet152
from easyai.model.backbone.cls.darknet import darknet21, darknet53
from easyai.model.backbone.cls.darknet import darknet21_dilated8, darknet21_dilated16
from easyai.model.backbone.cls.darknet import darknet53_dilated8, darknet53_dilated16
from easyai.model.backbone.cls.googlenet import GoogleNet
from easyai.model.backbone.cls.inceptionv4 import inceptionv4, inception_resnetv2
from easyai.model.backbone.cls.vgg import vgg13, vgg16, vgg19
from easyai.model.backbone.cls.squeezenet import SqueezeNet, DilatedSqueezeNet
from easyai.model.backbone.cls.densenet import densenet121, densenet201, densenet169, densenet161
from easyai.model.backbone.cls.densenet import densenet121_dilated8, densenet121_dilated16
from easyai.model.backbone.cls.senet import se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152
from easyai.model.backbone.cls.resnext import resnext50, resnext101, resnext152
from easyai.model.backbone.cls.nasnet import nasnet
from easyai.model.backbone.cls.preactresnet import preactresnet18, preactresnet34, preactresnet50
from easyai.model.backbone.cls.preactresnet import preactresnet101, preactresnet152
from easyai.model.backbone.cls.xception import xception
from easyai.model.backbone.cls.attention import attention56, attention92
from easyai.model.backbone.utility.my_backbone import MyBackbone
from easyai.model.utility.model_parse import ModelParse


class BackboneFactory():

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
        model = MyBackbone(model_define)
        model.set_name(file_name)
        return model

    def get_base_model_from_name(self, net_name):
        result = None
        if net_name == BackboneName.ShuffleNetV2_1_0:
            result = shufflenetv2_1_0()
        elif net_name == BackboneName.MobileNetV2_1_0:
            result = mobilenetv2_1_0()
        elif net_name == BackboneName.ResNet18:
            result = resnet18()
        elif net_name == BackboneName.ResNet34:
            result = resnet34()
        elif net_name == BackboneName.ResNet50:
            result = resnet50()
        elif net_name == BackboneName.ResNet101:
            result = resnet101()
        elif net_name == BackboneName.ResNet152:
            result = resnet152()
        elif net_name == BackboneName.Darknet53:
            result = darknet53()
        elif net_name == BackboneName.Darknet21:
            result = darknet21()
        elif net_name == BackboneName.Darknet21_Dilated8:
            result = darknet21_dilated8()
        elif net_name == BackboneName.Darknet21_Dilated16:
            result = darknet21_dilated16()
        elif net_name == BackboneName.Darknet53_Dilated8:
            result = darknet53_dilated8()
        elif net_name == BackboneName.Darknet53_Dilated16:
            result = darknet53_dilated16()
        elif net_name == BackboneName.GoogleNet:
            result = GoogleNet()
        elif net_name == BackboneName.InceptionV4:
            result = inceptionv4()
        elif net_name == BackboneName.InceptionResNetV2:
            result = inception_resnetv2()
        elif net_name == BackboneName.Vgg13:
            result = vgg13()
        elif net_name == BackboneName.Vgg16:
            result = vgg16()
        elif net_name == BackboneName.Vgg19:
            result = vgg19()
        elif net_name == BackboneName.SqueezeNet:
            result = SqueezeNet()
        elif net_name == BackboneName.DilatedSqueezeNet:
            result = DilatedSqueezeNet()
        elif net_name == BackboneName.Densenet121:
            result = densenet121()
        elif net_name == BackboneName.Densenet121_Dilated8:
            result = densenet121_dilated8()
        elif net_name == BackboneName.Densenet121_Dilated16:
            result = densenet121_dilated16()
        elif net_name == BackboneName.Densenet169:
            result = densenet169()
        elif net_name == BackboneName.Densenet201:
            result = densenet201()
        elif net_name == BackboneName.Densenet161:
            result = densenet161()
        elif net_name == BackboneName.SEResNet18:
            result = se_resnet18()
        elif net_name == BackboneName.SEResNet34:
            result = se_resnet34()
        elif net_name == BackboneName.SEResNet50:
            result = se_resnet50()
        elif net_name == BackboneName.SEResNet101:
            result = se_resnet101()
        elif net_name == BackboneName.SEResNet152:
            result = se_resnet152()
        elif net_name == BackboneName.ResNext50:
            result = resnext50()
        elif net_name == BackboneName.ResNext101:
            result = resnext101()
        elif net_name == BackboneName.ResNext152:
            result = resnext152()
        elif net_name == BackboneName.NasNet:
            result = nasnet()
        elif net_name == BackboneName.PreActResNet18:
            result = preactresnet18()
        elif net_name == BackboneName.PreActResNet34:
            result = preactresnet34()
        elif net_name == BackboneName.PreActResNet50:
            result = preactresnet50()
        elif net_name == BackboneName.PreActResNet101:
            result = preactresnet101()
        elif net_name == BackboneName.PreActResNet152:
            result = preactresnet152()
        elif net_name == BackboneName.Xception:
            result = xception()
        elif net_name == BackboneName.Attention56:
            result = attention56()
        elif net_name == BackboneName.Attention92:
            result = attention92()
        else:
            print("base model:%s error" % net_name)
        return result
