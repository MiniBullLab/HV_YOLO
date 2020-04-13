#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
import numpy as np
from easyai.helper import DirProcess


class SegmentSample():

    def __init__(self, train_path):
        self.train_path = train_path
        self.is_shuffled = False
        self.shuffled_vector = []
        self.image_and_label_list = []
        self.sample_count = 0
        self.annotation_post = ".png"
        self.dirProcess = DirProcess()

    def read_sample(self):
        self.image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        if self.is_shuffled:
            temp_index = index % self.sample_count
            temp_index = self.shuffled_vector[temp_index]
            img_path, label_path = self.image_and_label_list[temp_index]
        else:
            temp_index = index % self.sample_count
            img_path, label_path = self.image_and_label_list[temp_index]
        return img_path, label_path

    def get_sample_count(self):
        return len(self.image_and_label_list)

    def shuffle_sample(self):
        self.shuffled_vector = np.random.permutation(self.sample_count)
        self.is_shuffled = True

    def get_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../JPEGImages")
        labels_dir = os.path.join(path, "../SegmentLabel")
        for filename_and_post in self.dirProcess.getFileData(train_path):
            filename, post = os.path.splitext(filename_and_post)
            label_filename = filename + self.annotation_post
            label_path = os.path.join(labels_dir, label_filename)
            image_path = os.path.join(images_dir, filename_and_post)
            #print(image_path)
            if os.path.exists(label_path) and \
                    os.path.exists(image_path):
                result.append((image_path, label_path))
            else:
                print("%s or %s not exist" % (label_path, image_path))
        return result