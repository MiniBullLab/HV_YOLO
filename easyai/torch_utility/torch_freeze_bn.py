#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import re
import torch
import torch.nn as nn
from easyai.base_name.block_name import BlockType


class TorchFreezeNormalization():

    def __init__(self):
        pass

    def freeze_normalization_layer(self, model, layer_name, flag=0):
        if flag == 0:
            pass
        elif flag == 1:
            layer_name = layer_name.strip()
            self.freeze_front_layer(model, layer_name)
        elif flag == 2:
            layer_name = layer_name.strip()
            for key, block in model._modules.items():
                if key == BlockType.BaseNet:
                    self.freeze_front_layer(block, layer_name)
                    break
        elif flag == 3:
            layer_names = [x.strip() for x in layer_name.split(',') if x.strip()]
            self.freeze_layers(model, layer_names)
        elif flag == 4:
            layer_names = [x.strip() for x in layer_name.split(',') if x.strip()]
            for key, block in model._modules.items():
                if key == BlockType.BaseNet:
                    self.freeze_layers(block, layer_names)
                    break
        elif flag == 5:
            layer_name = layer_name.strip()
            self.freeze_layer_from_name(model, layer_name)
        elif flag == 6:
            layer_name = layer_name.strip()
            for key, block in model._modules.items():
                if key == BlockType.BaseNet:
                    self.freeze_layer_from_name(block, layer_name)
                    break
        else:
            print("freeze normalization error")

    def freeze_layers(self, model, layer_names):
        for key, block in model._modules.items():
            if key in layer_names:
                self.freeze_bn(block)

    def freeze_layer_from_name(self, model, layer_name):
        layer_name_re = None
        if layer_name is not None:
            layer_name_re = re.compile(layer_name)
        for key, block in model._modules.items():
            if layer_name_re.match(key) is not None:
                self.freeze_bn(block)

    def freeze_front_layer(self, model, layer_name):
        for key, block in model._modules.items():
            self.freeze_bn(block)
            if layer_name == key:
                break

    def freeze_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
