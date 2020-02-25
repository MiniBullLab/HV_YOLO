#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper import ImageProcess
from easyai.config import segment_config


class ConvertSegmentionLable():

    def __init__(self):
        self.save_label_dir = "SegmentLabel"
        self.annotation_post = ".png"
        self.dirProcess = DirProcess()
        self.image_process = ImageProcess()

    def create_segment_train(self, label_dir, is_gray, class_list):
        output_dir = os.path.join(label_dir, "../%s" % self.save_label_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        label_list = list(self.dirProcess.getDirFiles(label_dir, "*.*"))
        for label_path in label_list:
            path, file_name_and_post = os.path.split(label_path)
            print(label_path)
            if is_gray:
                mask = self.image_process.read_gray_image(label_path)
            else:
                _, mask = self.image_process.readRgbImage(label_path)
            if mask is not None:
                for index, class_name, value in enumerate(class_list):
                    if is_gray:
                        mask[mask == value] = index
                    else:
                        pass
                save_path = os.path.join(output_dir, file_name_and_post)
                cv2.imwrite(save_path, mask)