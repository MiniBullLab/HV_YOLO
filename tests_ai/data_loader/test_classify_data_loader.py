#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_classify_data_loader.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import json
import numpy as np
from easyai.data_loader.cls.classify_dataloader import get_classify_train_dataloader
from easyai.data_loader.cls.classify_dataloader import get_classify_val_dataloader
from easyai.visualization.task_show.classify_show import ClassifyShow
from easyai.config.utility.config_factory import ConfigFactory


def test_classify_train_data_loader(task_name, config_path, train_path, class_json):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = get_classify_train_dataloader(train_path,
                                               task_config.data_mean,
                                               task_config.data_std,
                                               task_config.image_size,
                                               task_config.image_channel,
                                               task_config.train_batch_size,
                                               task_config.train_data_augment)

    classify_show = ClassifyShow()
    file = open(class_json, 'r')
    class_name = json.load(file)

    img_num = 0
    for idx, (imgs, targets) in enumerate(dataloader):
        for img, target in zip(imgs, targets):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(img.numpy(), (1, 2, 0)).copy()
            target = str(target.numpy())
            classify_show.show(img, target,
                 class_name, scale=3.0)

def test_classify_val_data_loader(task_name, config_path, val_path, class_json):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(task_name, config_path=config_path)
    dataloader = get_classify_val_dataloader(val_path,
                                             task_config.data_mean,
                                             task_config.data_std,
                                             task_config.image_size,
                                             task_config.image_channel,
                                             task_config.test_batch_size)

    classify_show = ClassifyShow()
    file = open(class_json, 'r')
    class_name = json.load(file)

    img_num = 0
    for idx, (imgs, targets) in enumerate(dataloader):
        for img, target in zip(imgs, targets):
            print("img_num: {}".format(img_num))
            img_num += 1

            img = np.transpose(img.numpy(), (1, 2, 0)).copy()
            target = str(target.numpy())
            classify_show.show(img, target,
                 class_name, scale=3.0)


if __name__ == "__main__":
    task_name = "classify"
    config_path = ".log/config/classify_config.json"
    train_path = "/home/wfw/data/VOCdevkit/cifar100/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/cifar100/ImageSets/val.txt"
    class_json = "/home/wfw/data/VOCdevkit/cifar100/class.json"

    # test_classify_train_data_loader(task_name, config_path, train_path, class_json)
    test_classify_val_data_loader(task_name, config_path, val_path, class_json)