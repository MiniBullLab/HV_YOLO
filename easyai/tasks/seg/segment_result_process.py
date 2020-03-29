#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import numpy as np
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class SegmentResultProcess():

    def __init__(self):
        self.dataset_process = ImageDataSetProcess()

    def get_detection_result(self, prediction, threshold=0):
        result = None
        if prediction.ndim == 2:
            result = (prediction >= threshold).astype(int)
            # print(set(list(result.flatten())))
        elif prediction.ndim == 3:
            result = np.argmax(prediction, axis=0)
        elif prediction.ndim == 4:
            result = np.argmax(prediction, axis=1)
        return result

    def resize_segmention_result(self, src_size, image_size,
                                 segmention_result):
        ratio, pad = self.dataset_process.resize_square_size(src_size,
                                                             image_size)
        start_h = pad[1] // 2
        stop_h = image_size[1] - (pad[1] - (pad[1] // 2))
        start_w = pad[0] // 2
        stop_w = image_size[0] - (pad[0] - (pad[0] // 2))
        result = segmention_result[start_h:stop_h, start_w:stop_w]
        result = result.astype(np.float32)
        result = self.dataset_process.image_resize(result, src_size)
        return result

    def output_feature_map_resize(self, input_data, target):
        target = target.type(input_data.dtype)
        n, c, h, w = input_data.size()
        nt, ht, wt = target.size()
        # Handle inconsistent size between input and target
        if h > ht and w > wt:  # upsample labels
            target = target.unsequeeze(1)
            target = torch.nn.functional.upsample(target, size=(h, w), mode='nearest')
            target = target.sequeeze(1)
        elif h < ht and w < wt:  # upsample images
            input_data = torch.nn.functional.upsample(input_data, size=(ht, wt), mode='bilinear')
        elif h == ht and w == wt:
            pass
        else:
            print("input_data: (%d,%d) and target: (%d,%d) error "
                  % (h, w, ht, wt))
            raise Exception("segment_data_resize error")
        return input_data, target
