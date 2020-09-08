#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_segment_data_loader.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import numpy as np
from easyai.data_loader.sr.super_resolution_dataloader import get_sr_train_dataloader
from easyai.data_loader.sr.super_resolution_dataloader import get_sr_val_dataloader
from easyai.config.utility.config_factory import ConfigFactory
from easyai.visualization.sr_show import SuperResolutionShow


def test_sr_train_dataloader(task_name, config_path, train_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = get_sr_train_dataloader(train_path,
                                         task_config.image_size,
                                         task_config.image_channel,
                                         task_config.upscale_factor,
                                         task_config.train_batch_size)

    sr_show = SuperResolutionShow()

    img_num = 0
    for idx, (images, labels) in enumerate(dataloader):
        for image, label in zip(images, labels):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(image.numpy(), (1, 2, 0)).copy()
            sr_img = np.transpose(label.numpy(), (1, 2, 0)).copy()

            sr_show.show(img, sr_img, scale=0.5)


def test_sr_val_dataloader(task_name, config_path, val_path):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = get_sr_val_dataloader(val_path,
                                       task_config.image_size,
                                       task_config.image_channel,
                                       task_config.upscale_factor,
                                       task_config.test_batch_size)

    sr_show = SuperResolutionShow()

    img_num = 0
    for idx, (images, labels) in enumerate(dataloader):
        for image, label in zip(images, labels):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(image.numpy(), (1, 2, 0)).copy()
            sr_img = np.transpose(label.numpy(), (1, 2, 0)).copy()

            sr_show.show(img, sr_img, scale=0.5)


if __name__ == "__main__":
    task_name = "super_resolution"
    config_path = "../.log/config/super_resolution_config.json"
    train_path = "/home/wfw/data/VOCdevkit/DIV2k/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/DIV2k/ImageSets/val.txt"

    # test_sr_train_dataloader(task_name, config_path, train_path)
    test_sr_val_dataloader(task_name, config_path, val_path)