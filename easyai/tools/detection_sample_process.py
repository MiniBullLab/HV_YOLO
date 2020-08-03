#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess


class DetectionSampleProcess():

    def __init__(self):
        self.json_process = JsonProcess()
        self.dir_process = DirProcess()
        self.annotation_post = "*.json"

    def get_detection_class(self, train_path):
        all_names = []
        path, _ = os.path.split(train_path)
        annotation_dir = os.path.join(path, "../Annotations")
        for label_path in self.dir_process.getDirFiles(annotation_dir, self.annotation_post):
            print(label_path)
            _, boxes = self.json_process.parse_rect_data(label_path)
            temp_names = [box.name for box in boxes if box.name.strip()]
            all_names.extend(temp_names)
        class_names = set(all_names)
        return tuple(class_names)
