#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import codecs
import json
from easyai.config.base_config import BaseConfig


class ClassifyConfig(BaseConfig):

    def __init__(self):
        super().__init__()
        self.log_name = "classify"
        # data
        self.image_size = None  # W * H
        self.data_mean = None
        self.data_std = None
        # test
        self.test_batch_size = 1
        self.test_resul_name = 'cls_result.txt'
        self.test_result_path = os.path.join(self.root_save_dir, self.test_resul_name)

        # train
        self.train_batch_size = 1
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = None
        self.best_weights_name = None
        self.latest_weights_file = None
        self.best_weights_file = None
        self.max_epochs = 1
        self.base_lr = 0.1
        self.optimizer_config = None
        self.accumulated_batches = 1
        self.display = 1

        self.cls_config_dir = os.path.join(self.root_save_dir, self.config_save_dir)
        self.config_path = os.path.join(self.cls_config_dir, "classify.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()
        if self.snapshot_path is not None and not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, exist_ok=True)

    def load_config(self, config_path):
        if config_path is not None and os.path.exists(config_path):
            self.config_path = config_path
        if os.path.exists(self.config_path):
            with codecs.open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.load_data_value(config_dict)
            self.load_test_value(config_dict)
            self.load_train_value(config_dict)
        else:
            print("{} not exits".format(config_path))

    def save_config(self):
        if not os.path.exists(self.config_path):
            if not os.path.exists(self.cls_config_dir):
                os.makedirs(self.cls_config_dir, exist_ok=True)
        config_dict = {}
        self.save_data_value(config_dict)
        self.save_test_value(config_dict)
        self.save_train_value(config_dict)
        with codecs.open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, sort_keys=True, indent=4, ensure_ascii=False)

    def load_data_value(self, config_dict):
        if config_dict.get('image_size', None):
            self.image_size = tuple(config_dict['image_size'])
        if config_dict.get('data_mean', None):
            self.data_mean = tuple(config_dict['data_mean'])
        if config_dict.get('data_std', None):
            self.data_std = tuple(config_dict['data_std'])

    def save_data_value(self, config_dict):
        config_dict['image_size'] = self.image_size
        config_dict['data_mean'] = self.data_mean
        config_dict['data_std'] = self.data_std

    def load_test_value(self, config_dict):
        if config_dict.get('test_batch_size', None):
            self.test_batch_size = int(config_dict['test_batch_size'])

    def save_test_value(self, config_dict):
        config_dict['test_batch_size'] = self.test_batch_size

    def load_train_value(self, config_dict):
        if config_dict.get('train_batch_size', None):
            self.train_batch_size = int(config_dict['train_batch_size'])
        if config_dict.get('is_save_epoch_model', None):
            self.is_save_epoch_model = bool(config_dict['is_save_epoch_model'])
        if config_dict.get('latest_weights_name', None):
            self.latest_weights_name = str(config_dict['latest_weights_name'])
        if config_dict.get('best_weights_name', None):
            self.best_weights_name = str(config_dict['best_weights_name'])
        if config_dict.get('max_epochs', None):
            self.max_epochs = int(config_dict['max_epochs'])
        if config_dict.get('base_lr', None):
            self.base_lr = float(config_dict['base_lr'])
        if config_dict.get('optimizer_config', None):
            self.optimizer_config = config_dict['optimizer_config']
        if config_dict.get('accumulated_batches', None):
            self.accumulated_batches = int(config_dict['accumulated_batches'])
        if config_dict.get('display', None):
            self.display = int(config_dict['display'])

    def save_train_value(self, config_dict):
        config_dict['train_batch_size'] = self.train_batch_size
        config_dict['is_save_epoch_model'] = self.is_save_epoch_model
        config_dict['latest_weights_name'] = self.latest_weights_name
        config_dict['best_weights_name'] = self.best_weights_name
        config_dict['max_epochs'] = self.max_epochs
        config_dict['base_lr'] = self.base_lr
        config_dict['optimizer_config'] = self.optimizer_config
        config_dict['accumulated_batches'] = self.accumulated_batches
        config_dict['display'] = self.display

    def get_data_default_value(self):
        self.image_size = (32, 32)
        self.data_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.data_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    def get_test_default_value(self):
        self.test_batch_size = 1

    def get_train_default_value(self):
        self.train_batch_size = 64
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'cls_latest.pt'
        self.best_weights_name = 'cls_best.pt'

        self.latest_weights_file = os.path.join(self.snapshot_path, self.latest_weights_name)
        self.best_weights_file = os.path.join(self.snapshot_path, self.best_weights_name)

        self.max_epochs = 200

        self.base_lr = 0.1
        self.optimizer_config = {0: {'optimizer': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.accumulated_batches = 1
        self.display = 20
