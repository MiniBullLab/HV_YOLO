#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.model_name import ModelName
from easyai.model.utility.model_parse import ModelParse
from easyai.model.utility.my_model import MyModel
from easyai.model.det2d.yolov3_det2d import YoloV3Det2d
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

from easyai.model.utility.registry import REGISTERED_CLS_MODEL
from easyai.utility.registry import build_from_cfg
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

    def get_model_from_name(self, model_name, default_args=None):
        if default_args is None:
            default_args = {"data_channel": 3}
        if REGISTERED_CLS_MODEL.get(model_name):
            model = self.get_cls_model(model_name, default_args)
        elif model_name == ModelName.YoloV3Det2d:
            model = YoloV3Det2d(**default_args)
        elif model_name == ModelName.FCNSeg:
            model = FCN8sSeg(**default_args)
        elif model_name == ModelName.UNetSeg:
            model = UNetSeg(**default_args)
        elif model_name == ModelName.RefineNetSeg:
            model = RefineNetSeg(**default_args)
        elif model_name == ModelName.PSPNetSeg:
            model = PSPNetSeg(**default_args)
        elif model_name == ModelName.EncNetSeg:
            model = EncNetSeg(**default_args)
        elif model_name == ModelName.BiSeNet:
            model = BiSeNet(**default_args)
        elif model_name == ModelName.FastSCNN:
            model = FastSCNN(**default_args)
        elif model_name == ModelName.ICNet:
            model = ICNet(**default_args)
        elif model_name == ModelName.DeepLabV3:
            model = DeepLabV3(**default_args)
        elif model_name == ModelName.DeepLabV3Plus:
            model = DeepLabV3Plus(**default_args)
        elif model_name == ModelName.MobilenetDeepLabV3Plus:
            model = MobilenetDeepLabV3Plus(**default_args)
        elif model_name == ModelName.MobileV2FCN:
            model = MobileV2FCN(**default_args)
        elif model_name == ModelName.ComplexYOLO:
            model = ComplexYOLO(**default_args)
        else:
            model = self.get_sr_model(model_name, default_args)
        return model

    def get_cls_model(self, model_name, default_args):
        default_args['type'] = model_name.strip()
        model = build_from_cfg(default_args, REGISTERED_CLS_MODEL)
        return model

    def get_sr_model(self, model_name, default_args):
        model = None
        if model_name == ModelName.MSRResNet:
            model = MSRResNet(**default_args)
        elif model_name == ModelName.SmallSRNet:
            model = SmallSRNet(**default_args)
        return model
