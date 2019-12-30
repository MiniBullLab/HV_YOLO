#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from converter.keras_models.model_name import KerasModelName
from converter.keras_models.fgsegnetv2 import FgSegNetV2
from converter.keras_models.my_fgsegnetv2 import MyFgSegNetV2
from converter.keras_models.instance_normalization import InstanceNormalization
from converter.keras_models.my_upsampling_2d import MyUpSampling2D
from converter.keras_models.fgsegnetv2 import acc, loss
from converter.keras_models.model_process import KerasModelProcess
from keras import models


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
        else:
            print("base model:%s error" % net_name)
        return result

    def load_model(self, h5_model_path, net_name=None):
        result = None
        if net_name is None:
            result = models.load_model(h5_model_path)
        else:
            net_name = net_name.strip()
            if net_name in KerasModelName.FgSegNetV2:
                custom_layers = {'InstanceNormalization': InstanceNormalization,
                                 'MyUpSampling2D': MyUpSampling2D,
                                 'loss': loss, 'acc': acc}
                result = models.load_model(h5_model_path, custom_objects=custom_layers)
                # result = self.model_process.change_fgsegnet_layer(result)
            elif net_name in KerasModelName.MyFgSegNetV2:
                custom_layers = {'loss': loss, 'acc': acc}
                result = models.load_model(h5_model_path, custom_objects=custom_layers)
                # result = self.model_process.change_my_fgsegnet_layer(result)
            else:
                print("base model:%s error" % net_name)
        return result
