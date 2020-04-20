#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import numpy as np
from easyai.helper import DirProcess
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName


class ComputeImagesMean():

    def __init__(self, image_size):
        self.image_size = image_size
        self.dir_process = DirProcess()
        self.image_process = ImageProcess()
        self.dataset_process = ImageDataSetProcess()

    def compute(self, train_path):
        numpy_images = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../JPEGImages")
        for line_data in self.dir_process.getFileData(train_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) >= 1:
                image_path = os.path.join(images_dir, data_list[0])
                src_image, rgb_image = self.image_process.readRgbImage(image_path)
                rgb_image = self.dataset_process.image_resize(rgb_image, self.image_size)
                normaliza_image = self.dataset_process.image_normaliza(rgb_image)
                numpy_images.append(normaliza_image)
            else:
                print("read %s image path error!" % data_list)
        numpy_images = np.stack(numpy_images)
        mean = np.mean(numpy_images, axis=(0, 1, 2))
        std = np.std(numpy_images, axis=(0, 1, 2))
        return mean, std


def main():
    print("start...")
    options = ToolArgumentsParse.images_path_parse()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(options.task_name, config_path=options.config_path)
    test = ComputeImagesMean(image_size=task_config.image_size)
    mean, std = test.compute(options.inputPath)
    print(mean, std)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()
