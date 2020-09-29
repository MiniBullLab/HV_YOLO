#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
from easyai.model.backbone.utility.my_backbone import MyBackbone
from easyai.model.utility.model_parse import ModelParse
from easyai.model.backbone.utility.registry import REGISTERED_CLS_BACKBONE
from easyai.utility.registry import build_from_cfg


class BackboneFactory():

    def __init__(self):
        self.cfgReader = ModelParse()

    def get_backbone_model(self, model_config):
        input_name = model_config['type'].strip()
        model_args = model_config.copy()
        if input_name.endswith("cfg"):
            result = self.get_backbone_from_cfg(input_name)
        else:
            result = self.get_backbone_from_name(model_args)
            if result is None:
                print("backbone:%s error" % input_name)
        return result

    def get_backbone_from_cfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.cfgReader.readCfgFile(cfg_path)
        model = MyBackbone(model_define)
        model.set_name(file_name)
        return model

    def get_backbone_from_name(self, model_config):
        input_name = model_config['type'].strip()
        if model_config.get('data_channel') is None:
            model_config['data_channel'] = 3
        result = None
        if REGISTERED_CLS_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_config, REGISTERED_CLS_BACKBONE)
        return result

