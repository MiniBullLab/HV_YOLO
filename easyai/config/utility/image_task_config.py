#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import codecs
import json
from easyai.config.utility.base_config import BaseConfig


class ImageTaskConfig(BaseConfig):

    def __init__(self):
        super().__init__()
        # data
        self.image_size = None  # W * H
        self.image_channel = 3
        self.data_mean = None
        self.data_std = None
        # test
        self.test_batch_size = 1
        # train
        self.train_batch_size = 1
        self.enable_mixed_precision = False
        self.max_epochs = 0
        self.base_lr = 0.0
        self.optimizer_config = None
        self.lr_scheduler_config = None

        self.config_path = None

        self.get_base_default_value()

        if self.root_save_dir is not None and not os.path.exists(self.root_save_dir):
            os.makedirs(self.root_save_dir, exist_ok=True)

        if self.snapshot_dir is not None and not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def load_config(self, config_path):
        if config_path is not None and os.path.exists(config_path):
            self.config_path = config_path
        if os.path.exists(self.config_path):
            with codecs.open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.load_data_value(config_dict)
            self.load_test_value(config_dict)
            self.load_train_value(config_dict)
        else:
            print("{} not exits".format(self.config_path))

    def save_config(self):
        if not os.path.exists(self.config_save_dir):
            os.makedirs(self.config_save_dir, exist_ok=True)
        config_dict = {}
        self.save_data_value(config_dict)
        self.save_test_value(config_dict)
        self.save_train_value(config_dict)
        with codecs.open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, sort_keys=True, indent=4, ensure_ascii=False)

    def load_data_value(self, config_dict):
        pass

    def load_test_value(self, config_dict):
        pass

    def load_train_value(self, config_dict):
        pass

    def save_data_value(self, config_dict):
        pass

    def save_test_value(self, config_dict):
        pass

    def save_train_value(self, config_dict):
        pass

