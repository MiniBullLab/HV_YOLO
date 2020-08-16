#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from easy_converter.keras_models.utility.keras_model_name import KerasModelName
from easy_converter.keras_models.utility.keras_model_factory import KerasModelFactory
from easy_converter.keras_models.utility.keras_model_process import KerasModelProcess


def print_model(h5_path, model_name):
    model_factory = KerasModelFactory()
    keras_model = model_factory.load_model(h5_path, model_name)
    print(keras_model.summary())
    for index, layer in enumerate(keras_model.layers):
        if layer is not None:
            print(index, layer.name, len(layer.get_weights()))


def main():
    print_model("mdl-weights-31-0.99850-0.00447.h5",
                KerasModelName.MyFgSegNetV2)


if __name__ == "__main__":
    main()