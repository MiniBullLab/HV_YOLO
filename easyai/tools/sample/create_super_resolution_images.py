#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import cv2
from easyai.helper import DirProcess
from easyai.helper import ImageProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName


class CreateSuperResolutionImages():

    def __init__(self):
        self.save_lr_dir = "LRImages"
        self.dirProcess = DirProcess()
        self.image_process = ImageProcess()
        self.dataset_process = ImageDataSetProcess()

    def create_lr_images(self, images_dir, upscale_factor):
        output_dir = os.path.join(images_dir, "../%s" % self.save_lr_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for image_path in self.dirProcess.getDirFiles(images_dir, "*.*"):
            path, file_name_and_post = os.path.split(image_path)
            print(image_path)
            src_image = self.image_process.opencvImageRead(image_path)
            if src_image is not None:
                shape = src_image.shape[:2]  # shape = [height, width]
                new_shape = (round(shape[1] / upscale_factor), round(shape[0] / upscale_factor))
                lr_image = self.dataset_process.cv_image_resize(src_image, new_shape)
                save_path = os.path.join(output_dir, file_name_and_post)
                cv2.imwrite(save_path, lr_image)


def main():
    print("start...")
    options = ToolArgumentsParse.dir_path_parse()
    test = CreateSuperResolutionImages()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.SuperResolution_Task, config_path=options.config_path)
    test.create_lr_images(options.inputPath,
                          task_config.upscale_factor)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()