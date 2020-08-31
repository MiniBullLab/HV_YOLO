#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.model.utility.model_parse import ModelParse
from easyai.model.utility.my_model import MyModel
from easyai.model.utility.registry import REGISTERED_CLS_MODEL
from easyai.model.utility.registry import REGISTERED_Det2D_MODEL
from easyai.model.utility.registry import REGISTERED_SEG_MODEL
from easyai.model.utility.registry import REGISTERED_SR_MODEL
from easyai.utility.registry import build_from_cfg

from easyai.model.utility.mode_weight_init import ModelWeightInit


class ModelFactory():

    def __init__(self):
        self.modelParse = ModelParse()
        self.model_weight_init = ModelWeightInit()

    def get_model(self, model_config):
        input_name = model_config['type'].strip()
        if input_name.endswith("cfg"):
            model_config.pop("type")
            result = self.get_model_from_cfg(input_name, model_config)
        else:
            result = self.get_model_from_name(model_config)
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

    def get_model_from_name(self, model_config):
        if model_config.get('data_channel') is None:
            model_config['data_channel'] = 3
        model_name = model_config['type'].strip()
        model_config['type'] = model_name
        if REGISTERED_CLS_MODEL.has_class(model_name):
            model = self.get_cls_model(model_config)
        elif REGISTERED_Det2D_MODEL.has_class(model_name):
            model = self.get_det2d_model(model_config)
        elif REGISTERED_SEG_MODEL.has_class(model_name):
            model = self.get_seg_model(model_config)
        elif REGISTERED_SR_MODEL.has_class(model_name):
            model = self.get_sr_model(model_config)
        else:
            model = None
        return model

    def get_cls_model(self, model_config):
        model = build_from_cfg(model_config, REGISTERED_CLS_MODEL)
        return model

    def get_det2d_model(self, model_config):
        model = build_from_cfg(model_config, REGISTERED_Det2D_MODEL)
        return model

    def get_seg_model(self, model_config):
        model = build_from_cfg(model_config, REGISTERED_Det2D_MODEL)
        return model

    def get_sr_model(self, model_config):
        model = build_from_cfg(model_config, REGISTERED_Det2D_MODEL)
        return model

