#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
import numpy as np
from helper import DirProcess


class ClassifySample():

    def __init__(self, train_path):
        self.train_path = train_path
        self.image_and_label_list = []
        self.sample_count = 0
        self.dirProcess = DirProcess()

    def read_sample(self):
        self.image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, label = self.image_and_label_list[temp_index]
        return img_path, label

    def get_sample_count(self):
        return len(self.image_and_label_list)

    def get_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../JPEGImages")
        for line_data in self.dirProcess.getFileData(train_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) == 2:
                image_path = os.path.join(images_dir, data_list[0])
                # print(image_path)
                if os.path.exists(image_path):
                    result.append((image_path, int(data_list[1])))
                else:
                    print("%s not exist" % image_path)
            else:
                print("% error" % line_data)
        return result
