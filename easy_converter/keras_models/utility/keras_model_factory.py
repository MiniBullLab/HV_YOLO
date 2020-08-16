#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from keras import models
from easy_converter.keras_models.utility.keras_model_name import KerasModelName
from easy_converter.keras_models.utility.keras_model_process import KerasModelProcess
from easy_converter.keras_models.seg.instance_normalization import InstanceNormalization
from easy_converter.keras_models.seg.my_upsampling_2d import MyUpSampling2D
from easy_converter.keras_models.seg.fgsegnetv2 import acc, loss
from easy_converter.keras_models.seg.fgsegnetv2 import FgSegNetV2
from easy_converter.keras_models.seg.my_fgsegnetv2 import MyFgSegNetV2
from easy_converter.keras_models.seg.unet import UNet


class KerasModelFactory():

    def __init__(self):
        self.model_process = KerasModelProcess()

    def get_model(self, net_name):
        result = None
        net_name = net_name.strip()
        if net_name in KerasModelName.FgSegNetV2:
            model = FgSegNetV2()
            result = model.init_model()
        elif net_name in KerasModelName.MyFgSegNetV2:
            model = MyFgSegNetV2()
            result = model.init_model()
        elif net_name in KerasModelName.UNet:
            model = UNet()
            result = model.init_unet()
        else:
            print("base model:%s error" % net_name)
        return result

    def load_model(self, h5_model_path, net_name=None):
        result = None
        if net_name is None:
            result = models.load_model(h5_model_path)
            print(net_name)
        else:
            net_name = net_name.strip()
            if net_name in KerasModelName.FgSegNetV2:
                custom_layers = {'InstanceNormalization': InstanceNormalization,
                                 'MyUpSampling2D': MyUpSampling2D,
                                 'loss': loss, 'acc': acc}
                result = models.load_model(h5_model_path, custom_objects=custom_layers)
            elif net_name in KerasModelName.MyFgSegNetV2:
                custom_layers = {'loss': loss, 'acc': acc}
                result = models.load_model(h5_model_path, custom_objects=custom_layers)
            else:
                print("base model:%s error" % net_name)
        return result
