#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
import scipy.misc as misc
from data_loader.utility.image_dataset_process import ImageDataSetProcess


class SegmentResultProcess():

    def __init__(self):
        self.dataset_process = ImageDataSetProcess()

    def get_detection_result(self, prediction):
        result = None
        if prediction.ndim == 2:
            result = prediction
        elif prediction.ndim == 3:
            result = np.max(prediction, axis=1)
        return result

    def resize_segmention_result(self, src_image, image_size,
                                 segmention_result):
        image, ratio, pad = self.dataset_process.image_resize_square(src_image,
                                                                     image_size)
        start_h = pad[1] // 2
        stop_h = image_size[1] - (pad[1] - (pad[1] // 2))
        start_w = pad[0] // 2
        stop_w = image_size[0] - (pad[0] - (pad[0] // 2))
        result = segmention_result[start_h:stop_h, start_w:stop_w]
        result = result.astype(np.float32)
        result = misc.imresize(result, src_image.shape(), 'nearest',
                               mode='F')  # float32 with F mode
        return result
