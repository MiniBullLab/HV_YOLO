#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easy_convert.converter.tensorrt_utility import trt_helper
from easy_convert.converter.tensorrt_utility  import trt_int8_calibration_helper as int8_helper
import time
import numpy as np


def main():
    # Prepare a dataset for Calibration
    batch_size = 1
    img_size = (3, 128, 128)
    onnx_model_path = 'model_128.onnx'

    dataset = np.random.rand(1000, *img_size).astype(np.float32)
    max_batch_for_calibartion = 5
    transform = None

    # Prepare a stream
    calibration_stream = int8_helper.ImageBatchStreamDemo(dataset, transform, max_batch_for_calibartion, img_size)

    engine_model_path = "engine_int8.trt"
    engine_int8 = trt_helper.get_engine(batch_size, onnx_model_path, engine_model_path, fp16_mode=False, int8_mode=True,
                                        calibration_stream=calibration_stream, save_engine=True)
    assert engine_int8, 'Broken engine'
    context_int8 = engine_int8.create_execution_context()
    inputs_int8, outputs_int8, bindings_int8, stream_int8 = trt_helper.allocate_buffers(engine_int8)

    engine_model_path = "engine_int16.trt"
    engine_fp16 = trt_helper.get_engine(batch_size, onnx_model_path, engine_model_path, fp16_mode=True, int8_mode=False,
                                        save_engine=True)
    assert engine_fp16, 'Broken engine'
    context_fp16 = engine_fp16.create_execution_context()
    inputs_fp16, outputs_fp16, bindings_fp16, stream_fp16 = trt_helper.allocate_buffers(engine_fp16)

    engine_model_path = "engine.trt"
    engine = trt_helper.get_engine(batch_size, onnx_model_path, engine_model_path, fp16_mode=False, int8_mode=False,
                                   save_engine=True)
    assert engine, 'Broken engine'
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = trt_helper.allocate_buffers(engine)

    total_time_int8 = []
    total_time_fp16 = []
    total_time = []
    for i in range(1, dataset.shape[0]):
        x_input = dataset[i]
        inputs_int8[0].host = x_input.reshape(-1)

        tic_int8 = time.time()
        trt_helper.do_inference(context_int8, bindings=bindings_int8, inputs=inputs_int8, outputs=outputs_int8,
                                stream=stream_int8)
        toc_int8 = time.time()
        total_time_int8.append(toc_int8 - tic_int8)

        tic_fp16 = time.time()
        trt_helper.do_inference(context_fp16, bindings=bindings_fp16, inputs=inputs_fp16, outputs=outputs_fp16,
                                stream=stream_fp16)
        toc_fp16 = time.time()
        total_time_fp16.append(toc_fp16 - tic_fp16)

        tic = time.time()
        trt_helper.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        toc = time.time()
        total_time.append(toc - tic)

    print('Toal time used by engine_int8: {}'.format(np.mean(total_time_int8)))
    print('Toal time used by engine_fp16: {}'.format(np.mean(total_time_fp16)))
    print('Toal time used by engine: {}'.format(np.mean(total_time)))


if __name__ == '__main__':
    main()

