#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os

root_save_dir = "./log"
model_save_dir = "snapshot"

image_size = (512, 440)
train_batch_size = 1
maxEpochs = 100
lr = 1e-4

snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "seg_latest.h5")
best_weights_file = os.path.join(snapshotPath, "seg_best.h5")
save_image_dir = os.path.join(root_save_dir, "seg_img")

