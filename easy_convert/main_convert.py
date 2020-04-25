#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie


from easy_convert.converter.onnx_convert_caffe import OnnxConvertCaffe
from easy_convert.converter.onnx_convert_tensorflow import OnnxConvertTensorflow


def model_convert(task_name, onnx_path):
    if task_name.strip() == "DeNET":
        caffe_converter = OnnxConvertCaffe(onnx_path)
        caffe_converter.convert_caffe()
    elif task_name.strip() == "SegNET":
        pb_converter = OnnxConvertTensorflow(onnx_path)
        pb_converter.convert_tensorflow()
    else:
        print("input task error!")
