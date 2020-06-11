#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.model_name import ModelName
from easyai.model.utility.model_parse import ModelParse
from easyai.model.utility.my_model import MyModel
from easyai.model.cls.vgg_cls import VggNetCls
from easyai.model.cls.inceptionv4_cls import Inceptionv4Cls
from easyai.model.cls.senet_cls import SENetCls
from easyai.model.cls.ghostnet_cls import GhostNetCls
from easyai.model.sr.msr_resnet import MSRResNet
from easyai.model.sr.small_srnet import SmallSRNet
from easyai.model.seg.fcn_seg import FCN8sSeg
from easyai.model.seg.unet_seg import UNetSeg
from easyai.model.seg.refinenet_seg import RefineNetSeg
from easyai.model.seg.pspnet_seg import PSPNetSeg
from easyai.model.seg.encnet_seg import EncNetSeg
from easyai.model.seg.bisenet_seg import BiSeNet
from easyai.model.seg.fast_scnn_seg import FastSCNN
from easyai.model.seg.icnet_seg import ICNet
from easyai.model.seg.deeplabv3 import DeepLabV3
from easyai.model.seg.deeplabv3_plus import DeepLabV3Plus
from easyai.model.seg.mobilenet_deeplabv3_plus import MobilenetDeepLabV3Plus
from easyai.model.seg.mobilev2_fcn_seg import MobileV2FCN
from easyai.model.det3d.complex_yolo import ComplexYOLO

from easyai.model.utility.mode_weight_init import ModelWeightInit


class ModelFactory():

    def __init__(self):
        self.modelParse = ModelParse()
        self.model_weight_init = ModelWeightInit()

    def get_model(self, input_name, default_args=None):
        input_name = input_name.strip()
        if input_name.endswith("cfg"):
            result = self.get_model_from_cfg(input_name, default_args)
        else:
            result = self.get_model_from_name(input_name, default_args)
            if result is None:
                print("%s model error!" % input_name)
        self.model_weight_init.init_weight(result)
        return result

    def get_model_from_cfg(self, cfg_path, default_args=None):
        if not cfg_path.endswith("cfg"):
            print("%s model error" % cfg_path)
            return None
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.modelParse.readCfgFile(cfg_path)
        model = MyModel(model_define, path, default_args)
        model.set_name(file_name)
        return model

    def get_model_from_name(self, modelName, default_args=None):
        model = None
        if modelName == ModelName.VggNetCls:
            model = VggNetCls(**default_args)
        elif modelName == ModelName.Inceptionv4Cls:
            model = Inceptionv4Cls(**default_args)
        elif modelName == ModelName.SENetCls:
            model = SENetCls(**default_args)
        elif modelName == ModelName.GhostNetCls:
            model = GhostNetCls(**default_args)
        elif modelName == ModelName.MSRResNet:
            model = MSRResNet(**default_args)
        elif modelName == ModelName.SmallSRNet:
            model = SmallSRNet(**default_args)
        elif modelName == ModelName.FCNSeg:
            model = FCN8sSeg(**default_args)
        elif modelName == ModelName.UNetSeg:
            model = UNetSeg(**default_args)
        elif modelName == ModelName.RefineNetSeg:
            model = RefineNetSeg(**default_args)
        elif modelName == ModelName.PSPNetSeg:
            model = PSPNetSeg(**default_args)
        elif modelName == ModelName.EncNetSeg:
            model = EncNetSeg(**default_args)
        elif modelName == ModelName.BiSeNet:
            model = BiSeNet(**default_args)
        elif modelName == ModelName.FastSCNN:
            model = FastSCNN(**default_args)
        elif modelName == ModelName.ICNet:
            model = ICNet(**default_args)
        elif modelName == ModelName.DeepLabV3:
            model = DeepLabV3(**default_args)
        elif modelName == ModelName.DeepLabV3Plus:
            model = DeepLabV3Plus(**default_args)
        elif modelName == ModelName.MobilenetDeepLabV3Plus:
            model = MobilenetDeepLabV3Plus(**default_args)
        elif modelName == ModelName.MobileV2FCN:
            model = MobileV2FCN(**default_args)
        elif modelName == ModelName.ComplexYOLO:
            model = ComplexYOLO(**default_args)
        return model
