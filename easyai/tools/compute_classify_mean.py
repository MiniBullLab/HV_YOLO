#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import numpy as np
from easyai.helper.imageProcess import ImageProcess
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.config.task import classify_config


class ComputeClassifyMean():

    def __init__(self, train_path, image_size=classify_config.imgSize):
        self.image_size = image_size
        self.classify_sample = ClassifySample(train_path)
        self.classify_sample.read_sample(flag=0)
        self.image_process = ImageProcess()
        self.dataset_process = ImageDataSetProcess()

    def compute(self):
        numpy_images = []
        count = self.classify_sample.get_sample_count()
        for index in range(count):
            img_path, label = self.classify_sample.get_sample_path(index)
            src_image, rgb_image = self.image_process.readRgbImage(img_path)
            rgb_image = self.dataset_process.image_resize(rgb_image, self.image_size)
            normaliza_image = self.dataset_process.image_normaliza(rgb_image)
            numpy_images.append(normaliza_image)
        numpy_images = np.stack(numpy_images)
        mean = np.mean(numpy_images, axis=(0, 1, 2))
        std = np.std(numpy_images, axis=(0, 1, 2))
        return mean, std


def main():
    print("start...")
    test = ComputeClassifyMean("/home/lpj/github/dataset/ImageSets/train.txt")
    mean, std = test.compute()
    print(mean, std)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()
