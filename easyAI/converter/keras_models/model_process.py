#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import re
import numpy as np
import keras
from easyAI.converter.keras_models.model_name import KerasModelName
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Model


class KerasModelProcess():

    def __init__(self):
        pass

    def change_fgsegnet_layer(self, model):
        layers = [x for x in model.layers]
        x = layers[28].output
        a = layers[43].output
        b = layers[36].output
        for index in range(29, 32):
            if "instance_normalization" in layers[index].name:
                temp1 = BatchNormalization()
                x = temp1(x)
            elif "spatial_dropout2d" in layers[index].name:
                temp2 = Dropout(0.25)
                temp2.set_weights(layers[index].get_weights())
                x = temp2(x)
            else:
                x = layers[index](x)
        for index in range(32, 51):
            if "instance_normalization" in layers[index].name:
                temp1 = BatchNormalization()
                x = temp1(x)
            elif "multiply_1" in layers[index].name:
                x1 = layers[index]([x, b])
            elif "add_1" in layers[index].name:
                x = layers[index]([x, x1])
            elif "multiply_2" in layers[index].name:
                x2 = layers[index]([x, a])
            elif "add_2" in layers[index].name:
                x = layers[index]([x, x2])
            elif "conv2d_6" in layers[index].name:
                pass
            elif "global_average_pooling2d" in layers[index].name:
                pass
            else:
                x = layers[index](x)

        change_model = Model(inputs=model.input, outputs=x, name=KerasModelName.FgSegNetV2)
        return change_model

    def change_my_fgsegnet_layer(self, model):
        layers = [x for x in model.layers]
        x = layers[28].output
        a = layers[43].output
        b = layers[36].output
        for index in range(29, 32):
            if "spatial_dropout2d" in layers[index].name:
                temp2 = Dropout(0.25)
                temp2.set_weights(layers[index].get_weights())
                x = temp2(x)
            else:
                x = layers[index](x)
        for index in range(32, 51):
            if "multiply_1" in layers[index].name:
                x1 = layers[index]([x, b])
            elif "add_1" in layers[index].name:
                x = layers[index]([x, x1])
            elif "multiply_2" in layers[index].name:
                x2 = layers[index]([x, a])
            elif "add_2" in layers[index].name:
                x = layers[index]([x, x2])
            elif "conv2d_6" in layers[index].name:
                pass
            elif "global_average_pooling2d" in layers[index].name:
                pass
            else:
                x = layers[index](x)

        change_model = Model(inputs=model.input, outputs=x, name=KerasModelName.MyFgSegNetV2)
        return change_model

    def dropout2_replace(self, old_layer, **kwargs):
        temp2 = Dropout(0.25)
        temp2.set_weights(old_layer.get_weights())
        return temp2

    def make_list(self, X):
        if isinstance(X, list):
            return X
        return [X]

    def list_no_list(self, X):
        if len(X) == 1:
            return X[0]
        return X

    def replace_layer(self, model, replace_layer_subname, replacement_fn,
                      **kwargs):
        """
        args:
            model :: keras.models.Model instance
            replace_layer_subname :: str -- if str in layer name, replace it
            replacement_fn :: fn to call to replace all instances
                > fn output must produce shape as the replaced layers input
        returns:
            new model with replaced layers
        quick examples:
            want to just remove all layers with 'batch_norm' in the name:
                > new_model = replace_layer(model, 'batch_norm', lambda **kwargs : (lambda u:u))
            want to replace all Conv1D(N, m, padding='same') with an LSTM (lets say all have 'conv1d' in name)
                > new_model = replace_layer(model, 'conv1d', lambda layer, **kwargs: LSTM(units=layer.filters, return_sequences=True)
        """
        model_inputs = []
        model_outputs = []
        tsr_dict = {}

        model_output_names = [out.name for out in self.make_list(model.output)]

        for i, layer in enumerate(model.layers):
            ### Loop if layer is used multiple times
            for j in range(len(layer._inbound_nodes)):

                ### check layer inp/outp
                inpt_names = [inp.name for inp in self.make_list(layer.get_input_at(j))]
                outp_names = [out.name for out in self.make_list(layer.get_output_at(j))]

                ### setup model inputs
                if 'input' in layer.name:
                    for inpt_tsr in self.make_list(layer.get_output_at(j)):
                        model_inputs.append(inpt_tsr)
                        tsr_dict[inpt_tsr.name] = inpt_tsr
                    continue

                ### setup layer inputs
                inpt = self.list_no_list([tsr_dict[name] for name in inpt_names])

                ### remake layer
                if replace_layer_subname in layer.name:
                    print('replacing ' + layer.name)
                    x = replacement_fn(old_layer=layer, **kwargs)(inpt)
                else:
                    x = layer(inpt)

                ### reinstantialize outputs into dict
                for name, out_tsr in zip(outp_names, self.make_list(x)):

                    ### check if is an output
                    if name in model_output_names:
                        model_outputs.append(out_tsr)
                    tsr_dict[name] = out_tsr

        return Model(model_inputs, model_outputs)

    def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                            insert_layer_name=None, position='after'):

        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in model.layers:
            for node in layer.outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

        # Iterate over all layers after the input
        for layer in model.layers[1:]:

            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name):
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                new_layer = insert_layer_factory()
                if insert_layer_name:
                    new_layer.name = insert_layer_name
                else:
                    new_layer.name = '{}_{}'.format(layer.name,
                                                    new_layer.name)
                x = new_layer(x)
                print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                                layer.name))
                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

        return Model(inputs=model.inputs, outputs=x)
