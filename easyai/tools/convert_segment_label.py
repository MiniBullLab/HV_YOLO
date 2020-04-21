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
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName


class ConvertSegmentionLable():

    def __init__(self):
        self.save_label_dir = "SegmentLabel"
        self.annotation_post = ".png"
        self.dirProcess = DirProcess()
        self.image_process = ImageProcess()

    def convert_segment_label(self, label_dir, is_gray, class_list):
        output_dir = os.path.join(label_dir, "../%s" % self.save_label_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        label_list = list(self.dirProcess.getDirFiles(label_dir, "*.*"))
        for label_path in label_list:
            path, file_name_and_post = os.path.split(label_path)
            print(label_path)
            mask = self.process_segment_label(label_path, is_gray, class_list)
            if mask is not None:
                save_path = os.path.join(output_dir, file_name_and_post)
                cv2.imwrite(save_path, mask)

    def process_segment_label(self, label_path, is_gray, class_list):
        if is_gray:
            mask = self.image_process.read_gray_image(label_path)
        else:
            _, mask = self.image_process.readRgbImage(label_path)
        if mask is not None:
            if is_gray:
                mask = self.convert_gray_label(mask, class_list)
            else:
                mask = self.convert_color_label(mask, class_list)
        return mask

    def convert_gray_label(self, mask, class_list):
        shape = mask.shape  # shape = [height, width]
        result = np.full(shape, 250, dtype=np.uint8)
        for index, value in enumerate(class_list):
            gray_value = int(value[1].strip())
            result[mask == gray_value] = index
        return result

    def convert_color_label(self, mask, class_list):
        shape = mask.shape[:2]  # shape = [height, width]
        result = np.full(shape, 250, dtype=np.uint8)
        for index, value in enumerate(class_list):
            value_list = [int(x) for x in value[1].spilt(',') if x.strip()]
            color_value = np.array(value_list, dtype=np.uint8)
            temp1 = mask[:, :] == color_value
            temp2 = np.sum(temp1, axis=2)
            result[temp2 == 3] = index
        return result


def main():
    print("start...")
    options = ToolArgumentsParse.dir_path_parse()
    test = ConvertSegmentionLable()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.Segment_Task, config_path=options.config_path)
    test.convert_segment_label(options.inputPath,
                               task_config.label_is_gray,
                               task_config.class_name)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
