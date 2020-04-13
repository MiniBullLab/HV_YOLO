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
from easyai.model.sr.MSRResNet import MSRResNet
from easyai.model.sr.MySRModel import MySRModel
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
# pc
from easyai.model.pc_cls.pointnet_cls import PointNetCls
from easyai.model.pc_cls.pointnet2_cls import PointNet2Cls
from easyai.model.pc_seg.pointnet_seg import PointNetSeg
from easyai.model.pc_seg.pointnet2_seg import PointNet2Seg


class ModelFactory():

    def __init__(self):
        self.modelParse = ModelParse()

    def get_model(self, input_name, **kwargs):
        input_name = input_name.strip()
        if input_name.endswith("cfg"):
            result = self.get_model_from_cfg(input_name)
        else:
            result = self.get_model_from_name(input_name)
            if result is None:
                result = self.get_pc_model_from_name(input_name)
            if result is None:
                print("%s model error!" % input_name)
        return result

    def get_model_from_cfg(self, cfg_path, **kwargs):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.modelParse.readCfgFile(cfg_path)
        model = MyModel(model_define, path)
        model.set_name(file_name)
        return model

    def get_model_from_name(self, modelName, **kwargs):
        model = None
        if modelName == ModelName.VggNetCls:
            model = VggNetCls()
        elif modelName == ModelName.Inceptionv4Cls:
            model = Inceptionv4Cls()
        elif modelName == ModelName.SENetCls:
            model = SENetCls()
        elif modelName == ModelName.GhostNetCls:
            model = GhostNetCls()
        elif modelName == ModelName.MSRResNet:
            model = MSRResNet(in_nc=1, upscale_factor=3)
        elif modelName == ModelName.MySRModel:
            model = MySRModel(upscale_factor=3)
        elif modelName == ModelName.FCNSeg:
            model = FCN8sSeg()
        elif modelName == ModelName.UNetSeg:
            model = UNetSeg()
        elif modelName == ModelName.RefineNetSeg:
            model = RefineNetSeg()
        elif modelName == ModelName.PSPNetSeg:
            model = PSPNetSeg()
        elif modelName == ModelName.EncNetSeg:
            model = EncNetSeg()
        elif modelName == ModelName.BiSeNet:
            model = BiSeNet()
        elif modelName == ModelName.FastSCNN:
            model = FastSCNN()
        elif modelName == ModelName.ICNet:
            model = ICNet()
        elif modelName == ModelName.DeepLabV3:
            model = DeepLabV3()
        elif modelName == ModelName.DeepLabV3Plus:
            model = DeepLabV3Plus()
        elif modelName == ModelName.MobilenetDeepLabV3Plus:
            model = MobilenetDeepLabV3Plus()
        elif modelName == ModelName.MobileV2FCN:
            model = MobileV2FCN()
        elif modelName == ModelName.ComplexYOLO:
            model = ComplexYOLO()
        return model

    def get_pc_model_from_name(self, modelName, **kwargs):
        model = None
        if modelName == ModelName.PointNetCls:
            model = PointNetCls()
        elif modelName == ModelName.PointNet2Cls:
            model = PointNet2Cls()
        elif modelName == ModelName.PointNetSeg:
            model = PointNetSeg()
        elif modelName == ModelName.PointNet2Seg:
            model = PointNet2Seg()
        return model
