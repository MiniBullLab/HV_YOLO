#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_det2d_seg_data_loader.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import cv2
import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.multi_task.det2d_seg_train_dataloader import Det2dSegTrainDataloader
from easyai.data_loader.multi_task.det2d_seg_val_dataloader import get_det2d_seg_val_dataloader
from easyai.visualization.task_show.det2d_seg_drawing import Det2dSegTaskShow
from easyai.config.utility.config_factory import ConfigFactory


def decode_labels(img, labels):
    h, w, _ = img.shape

    x1 = w * (labels[1] - labels[3] / 2)
    y1 = h * (labels[2] - labels[4] / 2)
    x2 = w * (labels[1] + labels[3] / 2)
    y2 = h * (labels[2] + labels[4] / 2)

    return x1, y1, x2, y2

def test_det2d_seg_train_data_loader(task_name, config_path, train_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = Det2dSegTrainDataloader(train_path,
                                         task_config.detect_name,
                                         task_config.segment_name,
                                         task_config.train_batch_size,
                                         task_config.image_size,
                                         task_config.image_channel,
                                         multi_scale=task_config.train_multi_scale,
                                         is_augment=task_config.train_data_augment,
                                         balanced_sample=task_config.balanced_sample)
    det2d_seg_show = Det2dSegTaskShow()

    img_num = 0
    for i, (images, detects, segments) in enumerate(dataloader):
        for image, detect, segment in zip(images, detects, segments):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(image.numpy(), (1, 2, 0)).copy()
            target = detect.numpy()
            segment = segment.numpy()
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

            det2d_seg_show.show(img, segment,
                                task_config.label_is_gray,
                                task_config.segment_name,
                                results, scale=1.0)


if __name__ == "__main__":
    task_name = "det2d_seg"
    config_path = "../.log/config/det2d_seg_config_tusimple.json"
    train_path = "/home/wfw/data/VOCdevkit/Tusimple/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/Tusimple/ImageSets/val.txt"

    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)

    test_det2d_seg_train_data_loader(task_name, config_path, train_path)
    # test_detect2d_val_data_loader(task_name, config_path, val_path)