#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from flashtorch.activmax import GradientAscent

from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_model_process import TorchModelProcess


class ModelActivationMaxvisualize():

    def __init__(self, input_size=(352, 640), data_channel=3):
        self.input_size = input_size
        self.data_channel = data_channel
        self.model_factory = ModelFactory()
        self.model_process = TorchModelProcess()

    def show_convolutional_layer(self, model_name, weight_path, list_conv_index):
        model = self.model_factory.get_model(model_name,
                                             default_args={"data_channel": self.data_channel})
        self.model_process.loadLatestModelWeight(weight_path, model)
        list(model.features.named_children())
        g_ascent = GradientAscent(model.features)
        for index in list_conv_index:
            conv_layer = model.features[index]
            g_ascent.visualize(conv_layer, title='Randomly selected filters from conv_%d' % index)
            output = g_ascent.visualize(conv_layer, 3, title='conv_%d filter 3' % index, return_output=True)
            print('num_iter:', len(output))
            print('optimized image:', output[-1].shape)

            g_ascent.deepdream('./images/jay.jpg', conv_layer, 33)
