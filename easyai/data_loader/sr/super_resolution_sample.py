#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
import numpy as np
from easyai.helper import DirProcess


class SuperResolutionSample():

    def __init__(self, train_path):
        self.train_path = train_path
        self.is_shuffled = False
        self.shuffled_vector = []
        self.lr_and_hr_list = []
        self.sample_count = 0
        self.dirProcess = DirProcess()

    def read_sample(self):
        self.lr_and_hr_list = self.get_lr_and_hr_list(self.train_path)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        if self.is_shuffled:
            temp_index = index % self.sample_count
            temp_index = self.shuffled_vector[temp_index]
            img_path, label_path = self.lr_and_hr_list[temp_index]
        else:
            temp_index = index % self.sample_count
            img_path, label_path = self.lr_and_hr_list[temp_index]
        return img_path, label_path

    def get_sample_count(self):
        return len(self.lr_and_hr_list)

    def shuffle_sample(self):
        self.shuffled_vector = np.random.permutation(self.sample_count)
        self.is_shuffled = True

    def get_lr_and_hr_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        lr_dir = os.path.join(path, "../LRImages")
        hr_dir = os.path.join(path, "../HRImages")
        for filename_and_post in self.dirProcess.getFileData(train_path):
            # filename, post = os.path.splitext(filename_and_post)
            lr_path = os.path.join(lr_dir, filename_and_post)
            hr_path = os.path.join(hr_dir, filename_and_post)
            # print(lr_path)
            if os.path.exists(hr_path) and \
                    os.path.exists(lr_path):
                result.append((lr_path, hr_path))
            else:
                print("%s or %s not exist" % (lr_path, hr_path))
        return result
