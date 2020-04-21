#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import shutil
from easyai.helper.dirProcess import DirProcess


class CopyImage():

    def __init__(self):
        self.dir_process = DirProcess()

    def copy(self, trainPath, image_save_dir):
        image_list = self.get_image_list(trainPath)
        if os.path.exists(image_save_dir):
            os.system('rm -rf ' + image_save_dir)
        os.makedirs(image_save_dir, exist_ok=True)
        if len(image_list) > 0:
            image_path = image_list[0]
            path, image_name = os.path.split(image_path)
            save_path = os.path.join(image_save_dir, image_name)
            shutil.copy(image_path, save_path)
        else:
            print("empty images")

    def get_image_list(self, trainPath):
        result = []
        path, _ = os.path.split(trainPath)
        imagesDir = os.path.join(path, "../JPEGImages")
        for fileNameAndPost in self.dir_process.getFileData(trainPath):
            imagePath = os.path.join(imagesDir, fileNameAndPost)
            #print(imagePath)
            if os.path.exists(imagePath):
                result.append(imagePath)
        return result
