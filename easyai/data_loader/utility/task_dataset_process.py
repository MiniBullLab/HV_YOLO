#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class TaskDataSetProcess(BaseDataSetProcess):

    def __init__(self, resize_type, normalize_type, mean, std, pad_color):
        super().__init__()
        self.resize_type = resize_type
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pad_color = pad_color
        self.dataset_process = ImageDataSetProcess()