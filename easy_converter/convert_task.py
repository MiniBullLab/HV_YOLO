#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easy_converter.converter.onnx_convert_caffe import OnnxConvertCaffe
from easy_converter.converter.onnx_convert_tensorflow import OnnxConvertTensorflow


def easy_model_convert(task_name, onnx_path):
    if (onnx_path is None) or (not os.path.exists(onnx_path)):
        print("%s model not exists!" % onnx_path)
        return
    if task_name.strip() == "DeNET":
        caffe_converter = OnnxConvertCaffe(onnx_path)
        caffe_converter.convert_caffe()
    elif task_name.strip() == "SegNET":
        pb_converter = OnnxConvertTensorflow(onnx_path)
        pb_converter.convert_tensorflow()
    else:
        print("input task error!")
