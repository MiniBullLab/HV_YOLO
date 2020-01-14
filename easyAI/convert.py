#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

from easyAI.config import detectConfig
from easyAI.config import fgseg_config
from easyAI.tools.onnx_convert_caffe import OnnxConvertCaffe
from easyAI.converter.keras_models.model_name import KerasModelName
from easyAI.tools.keras_convert_tensorflow import KerasConvertTensorflow


def model_convert(task_name):
    if task_name == "DeNET":
        caffe_converter = OnnxConvertCaffe(detectConfig.save_onnx_path)
        caffe_converter.convert_caffe()
    elif task_name == "SegNET":
        pb_converter = KerasConvertTensorflow(fgseg_config.best_weights_file,
                                              KerasModelName.MyFgSegNetV2)
        pb_converter.keras_convert_tensorflow()
    else:
        print("error input")
