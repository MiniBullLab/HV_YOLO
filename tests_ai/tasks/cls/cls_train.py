#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: detect2d_test.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.cls.classify_train import ClassifyTrain


class ClsTrainTask():

    def __init__(self, train_path, val_path, pretrain_model_path):
        self.train_path = train_path
        self.val_path = val_path
        self.pretrain_model_path = pretrain_model_path

    def classify_train(self, cfg_path, gpu_id, config_path):
        cls_train_task = ClassifyTrain(cfg_path, gpu_id, config_path)
        cls_train_task.load_pretrain_model(self.pretrain_model_path)
        cls_train_task.train(self.train_path, self.val_path)

def main(train_path, val_path, model, config_path, weights):
    print("process start...")
    train_task = ClsTrainTask(train_path, val_path, weights)
    train_task.classify_train(model, 0, config_path)
    print("process end!")


if __name__ == '__main__':
    train_path = "/home/wfw/data/VOCdevkit/CatDog_classify/ImageSets/train.txt"
    val_path = "/home/wfw/data/VOCdevkit/CatDog_classify/ImageSets/val.txt"
    model = "../cfg/cls/resnet_classify.cfg"
    config_path = "./tasks/cls/classify_config_224.json"
    weights = "/home/wfw/HASCO/all_wights/resnet18_224.pt"

    main(train_path, val_path, model, config_path, weights)
