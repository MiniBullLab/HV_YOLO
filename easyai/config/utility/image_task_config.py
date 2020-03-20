#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.config.utility.base_config import BaseConfig


class ImageTaskConfig(BaseConfig):

    def __init__(self):
        super().__init__()
        # data
        self.image_size = None  # W * H
        # test
        self.test_batch_size = 1
        # train
        self.train_batch_size = 1
        self.enable_mixed_precision = False
        self.max_epochs = 0
        self.base_lr = 0.0
        self.optimizer_config = None
        self.lr_scheduler_config = None