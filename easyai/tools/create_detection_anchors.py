#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import numpy as np
from scipy import cluster
from easyai.data_loader.det.detection_dataset_process import DetectionDataSetProcess
from easyai.data_loader.det.detection_sample import DetectionSample
from easyai.helper import XMLProcess
from easyai.helper import ImageProcess
from easyai.config.task import detect2d_config


class CreateDetectionAnchors():

    def __init__(self, train_path):
        self.xmlProcess = XMLProcess()
        self.image_process = ImageProcess()
        self.detection_sample = DetectionSample(train_path,
                                                detect2d_config.className)
        self.detection_sample.read_sample()
        self.dataset_process = DetectionDataSetProcess()

    def get_anchors(self, number):
        wh_numpy = self.get_width_height()
        # Kmeans calculation
        k = cluster.vq.kmeans(wh_numpy, number)[0]
        k = k[np.argsort(k.prod(1))]  # sort small to large
        # Measure IoUs
        iou = np.stack([self.compute_iou(wh_numpy, x) for x in k], 0)
        biou = iou.max(0)[0]  # closest anchor IoU
        print('Best possible recall: %.3f' % (biou > 0.2635).float().mean())  # BPR (best possible recall)

        # Print
        print('kmeans anchors (n=%g, img_size=%g, IoU=%.2f/%.2f/%.2f-min/mean/best): ' %
              (number, detect2d_config.imgSize, biou.min(), iou.mean(), biou.mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')

    def get_width_height(self):
        count = self.detection_sample.get_sample_count()
        result = []
        for index in range(count):
            img_path, label_path = self.detection_sample.get_sample_path(index)
            src_image, rgb_image = self.image_process.readRgbImage(img_path)
            _, _, boxes = self.xmlProcess.parseRectData(label_path)
            rgb_image, labels = self.dataset_process.resize_dataset(rgb_image,
                                                                    detect2d_config.imgSize,
                                                                    boxes,
                                                                    detect2d_config.className)
            temp = np.zeros((len(labels), 2), dtype=np.float32)
            for index, object in enumerate(labels):
                temp[index, :] = np.array([object.width(), object.height()])
            result.append(temp)
        return np.concatenate(result, axis=0)

    def compute_iou(self, list_x, x2):
        result = np.zeros((len(list_x), 1), dtype=np.float32)
        for index, x1 in enumerate(list_x):
            min_w = min(x1[0], x2[0])
            min_h = min(x1[0], x2[1])
            iou = (min_w * min_h) / (x1[0] * x1[1] + x2[0] * x2[1] - min_w * min_h)
            result[index] = iou
        return result


def test():
    print("start...")
    test = CreateDetectionAnchors("/home/lpj/github/data/Berkeley/ImageSets/train.txt")
    test.get_anchors(9)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()
