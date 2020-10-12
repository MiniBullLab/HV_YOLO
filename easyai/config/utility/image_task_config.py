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
        self.data_channel = 3
        self.resize_type = 0
        self.normalize_type = 0
        self.data_mean = (0, 0, 0)
        self.data_std = (1, 1, 1)
        self.save_result_path = None
        # test
        self.test_batch_size = 1
        self.evaluation_result_name = None
        self.evaluation_result_path = None

        self.get_base_default_value()

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

    def load_image_data_value(self, config_dict):
        if config_dict.get('image_size', None) is not None:
            self.image_size = tuple(config_dict['image_size'])
        if config_dict.get('data_channel', None) is not None:
            self.data_channel = int(config_dict['data_channel'])
        if config_dict.get('resize_type', None) is not None:
            self.resize_type = int(config_dict['resize_type'])
        if config_dict.get('normalize_type', None) is not None:
            self.normalize_type = int(config_dict['normalize_type'])
        if config_dict.get('data_mean', None) is not None:
            self.data_mean = tuple(config_dict['data_mean'])
        if config_dict.get('data_std', None) is not None:
            self.data_std = tuple(config_dict['data_std'])

    def save_image_data_value(self, config_dict):
        config_dict['image_size'] = self.image_size
        config_dict['data_channel'] = self.data_channel
        config_dict['resize_type'] = self.resize_type
        config_dict['normalize_type'] = self.normalize_type
        config_dict['data_mean'] = self.data_mean
        config_dict['data_std'] = self.data_std

    def load_data_value(self, config_dict):
        pass

    def save_data_value(self, config_dict):
        pass

    def load_test_value(self, config_dict):
        pass

    def save_test_value(self, config_dict):
        pass

    def load_train_value(self, config_dict):
        pass

    def save_train_value(self, config_dict):
        pass

