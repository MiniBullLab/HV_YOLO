#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
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

        self.get_base_default_value()

        if self.root_save_dir is not None and not os.path.exists(self.root_save_dir):
            os.makedirs(self.root_save_dir, exist_ok=True)

        if self.snapshot_path is not None and not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, exist_ok=True)

    def load_config(self, config_path):
        pass

    def save_config(self):
        if not os.path.exists(self.config_save_dir):
            os.makedirs(self.config_save_dir, exist_ok=True)
