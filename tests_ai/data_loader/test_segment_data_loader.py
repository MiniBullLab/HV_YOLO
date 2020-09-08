#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_segment_data_loader.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import numpy as np
from easyai.data_loader.seg.segment_dataloader import get_segment_train_dataloader
from easyai.data_loader.seg.segment_dataloader import get_segment_val_dataloader
from easyai.config.utility.config_factory import ConfigFactory
from easyai.visualization.task_show.segment_show import SegmentionShow


def test_segment_train_dataloader(task_name, config_path, train_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    number_class = len(task_config.class_name)
    dataloader = get_segment_train_dataloader(train_path,
                                              number_class,
                                              task_config.image_size,
                                              task_config.image_channel,
                                              task_config.train_batch_size,
                                              is_augment=task_config.train_data_augment)

    segmention_show = SegmentionShow()

    img_num = 0
    for i, (images, segments) in enumerate(dataloader):
        for image, segment in zip(images, segments):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(image.numpy(), (1, 2, 0)).copy()
            segment = segment.numpy()
            segmention_show.show(img, segment,
                                 task_config.label_is_gray,
                                 task_config.class_name,
                                 scale=1.0)

def test_segment_val_dataloader(task_name, config_path, val_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    number_class = len(task_config.class_name)
    dataloader = get_segment_val_dataloader(val_path, number_class,
                                            task_config.image_size,
                                            task_config.image_channel,
                                            task_config.test_batch_size)

    segmention_show = SegmentionShow()

    img_num = 0
    for i, (images, segments) in enumerate(dataloader):
        for image, segment in zip(images, segments):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(image.numpy(), (1, 2, 0)).copy()
            segment = segment.numpy()
            segmention_show.show(img, segment,
                                 task_config.label_is_gray,
                                 task_config.class_name,
                                 scale=1.0)


if __name__ == "__main__":
    task_name = "segment"
    config_path = "../.log/config/segmention_config.json"
    train_path = "/home/wfw/data/VOCdevkit/CULane/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/CULane/ImageSets/val.txt"

    test_segment_train_dataloader(task_name, config_path, train_path)
    # test_segment_val_dataloader(task_name, config_path, train_path)