#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_detect_data_loader.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import cv2
import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.det2d.det2d_train_dataloader import DetectionTrainDataloader
from easyai.data_loader.det2d.det2d_val_dataloader import get_detection_val_dataloader
from easyai.visualization.task_show.detect2d_show import DetectionShow
from easyai.config.utility.config_factory import ConfigFactory


def decode_labels(img, labels):
    h, w, _ = img.shape

    x1 = w * (labels[1] - labels[3] / 2)
    y1 = h * (labels[2] - labels[4] / 2)
    x2 = w * (labels[1] + labels[3] / 2)
    y2 = h * (labels[2] + labels[4] / 2)

    return x1, y1, x2, y2

def test_detect2d_train_data_loader(task_name, config_path, train_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = DetectionTrainDataloader(train_path,
                                          task_config.class_name,
                                          task_config.train_batch_size,
                                          task_config.image_size,
                                          task_config.image_channel,
                                          multi_scale=task_config.train_multi_scale,
                                          is_augment=task_config.train_data_augment,
                                          balanced_sample=task_config.balanced_sample)

    detection_show = DetectionShow()

    img_num = 0
    for i, (images, targets) in enumerate(dataloader):
        for image, target in zip(images, targets):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(image.numpy(), (1, 2, 0)).copy()
            target = target.numpy()
            results = []
            for t in target:
                xmin, ymin, xmax, ymax = decode_labels(img, t)

                b = Rect2D()
                b.min_corner.x = xmin
                b.min_corner.y = ymin
                b.max_corner.x = xmax
                b.max_corner.y = ymax
                b.classIndex = int(t[0])

                results.append(b)

            detection_show.show(img, results, scale=1.0)

def test_detect2d_val_data_loader(task_name, config_path, val_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = get_detection_val_dataloader(val_path,
                                              task_config.class_name,
                                              image_size=task_config.image_size,
                                              data_channel=task_config.image_channel,
                                              batch_size=1)

    img_num = 0
    for i, (image_paths, src_images, input_images) in enumerate(dataloader):
        for image_path, src_image, input_image in zip(image_paths, src_images, input_images):
            print("img_num: {}, {}".format(img_num, image_path))
            img_num += 1

            cv2.imshow("src_image", src_image.numpy())
            cv2.waitKey(1)
            input_image = np.transpose(input_image.numpy(), (1, 2, 0)).copy()
            cv2.imshow("input_image", input_image)
            cv2.waitKey()


if __name__ == "__main__":
    task_name = "detect2d"
    config_path = "../.log/config/detection2d_config_berkeley.json"
    train_path = "/home/wfw/data/VOCdevkit/BKLdata/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/BKLdata/ImageSets/val.txt"

    test_detect2d_train_data_loader(task_name, config_path, train_path)
    # test_detect2d_val_data_loader(task_name, config_path, val_path)