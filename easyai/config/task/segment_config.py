#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import codecs
import json
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig


class SegmentionConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Segment_Task)
        # data
        self.label_is_gray = None
        self.class_name = None
        # test
        self.save_evaluation_path = os.path.join(self.root_save_dir, 'seg_evaluation.txt')
        # train
        self.log_name = "segment"
        self.train_data_augment = True
        self.is_save_epoch_model = False
        self.latest_weights_name = 'seg_latest.pt'
        self.best_weights_name = 'seg_best.pt'
        self.latest_weights_file = None
        self.best_weights_file = None
        self.accumulated_batches = 1
        self.display = 1

        self.freeze_layer_type = 0
        self.freeze_layer_name = None

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = None

        self.config_path = os.path.join(self.config_save_dir, "segmention_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

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
        super().save_config()
        config_dict = {}
        self.save_data_value(config_dict)
        self.save_test_value(config_dict)
        self.save_train_value(config_dict)
        with codecs.open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, sort_keys=True, indent=4, ensure_ascii=False)

    def load_data_value(self, config_dict):
        if config_dict.get('image_size', None) is not None:
            self.image_size = tuple(config_dict['image_size'])
        if config_dict.get('label_is_gray', None) is not None:
            self.label_is_gray = bool(config_dict['label_is_gray'])
        if config_dict.get('class_name', None) is not None:
            self.class_name = list(config_dict['class_name'])

    def save_data_value(self, config_dict):
        config_dict['image_size'] = self.image_size
        config_dict['label_is_gray'] = self.label_is_gray
        config_dict['class_name'] = self.class_name

    def load_test_value(self, config_dict):
        if config_dict.get('test_batch_size', None) is not None:
            self.test_batch_size = int(config_dict['test_batch_size'])

    def save_test_value(self, config_dict):
        config_dict['test_batch_size'] = self.test_batch_size

    def load_train_value(self, config_dict):
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])
        if config_dict.get('train_batch_size', None) is not None:
            self.train_batch_size = int(config_dict['train_batch_size'])
        if config_dict.get('is_save_epoch_model', None) is not None:
            self.is_save_epoch_model = bool(config_dict['is_save_epoch_model'])
        if config_dict.get('latest_weights_name', None) is not None:
            self.latest_weights_name = str(config_dict['latest_weights_name'])
        if config_dict.get('best_weights_name', None) is not None:
            self.best_weights_name = str(config_dict['best_weights_name'])
        if config_dict.get('max_epochs', None) is not None:
            self.max_epochs = int(config_dict['max_epochs'])
        if config_dict.get('base_lr', None) is not None:
            self.base_lr = float(config_dict['base_lr'])
        if config_dict.get('optimizer_config', None) is not None:
            optimizer_dict = config_dict['optimizer_config']
            self.optimizer_config = {}
            for epoch, value in optimizer_dict.items():
                self.optimizer_config[int(epoch)] = value
        if config_dict.get('lr_scheduler_config', None) is not None:
            self.lr_scheduler_config = config_dict['lr_scheduler_config']
        if config_dict.get('accumulated_batches', None) is not None:
            self.accumulated_batches = int(config_dict['accumulated_batches'])
        if config_dict.get('display', None) is not None:
            self.display = int(config_dict['display'])
        if config_dict.get('freeze_layer_type', None) is not None:
            self.freeze_layer_type = int(config_dict['freeze_layer_type'])
        if config_dict.get('freeze_layer_name', None) is not None:
            self.freeze_layer_name = config_dict['freeze_layer_name']
        if config_dict.get('freeze_bn_type', None) is not None:
            self.freeze_bn_type = int(config_dict['freeze_bn_type'])
        if config_dict.get('freeze_bn_layer_name', None) is not None:
            self.freeze_bn_layer_name = config_dict['freeze_bn_layer_name']

    def save_train_value(self, config_dict):
        config_dict['train_data_augment'] = self.train_data_augment
        config_dict['train_batch_size'] = self.train_batch_size
        config_dict['is_save_epoch_model'] = self.is_save_epoch_model
        config_dict['latest_weights_name'] = self.latest_weights_name
        config_dict['best_weights_name'] = self.best_weights_name
        config_dict['max_epochs'] = self.max_epochs
        config_dict['base_lr'] = self.base_lr
        config_dict['optimizer_config'] = self.optimizer_config
        config_dict['lr_scheduler_config'] = self.lr_scheduler_config
        config_dict['accumulated_batches'] = self.accumulated_batches
        config_dict['display'] = self.display
        config_dict['freeze_layer_type'] = self.freeze_layer_type
        config_dict['freeze_layer_name'] = self.freeze_layer_name
        config_dict['freeze_bn_type'] = self.freeze_bn_type
        config_dict['freeze_bn_layer_name'] = self.freeze_bn_layer_name

    def get_data_default_value(self):
        self.image_size = (500, 400)  # w * H
        self.label_is_gray = True
        self.class_name = [('background', '255'),
                           ('lane', '0')]

    def get_test_default_value(self):
        self.test_batch_size = 1

    def get_train_default_value(self):
        self.log_name = "segment"
        self.train_data_augment = False
        self.train_batch_size = 1
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'seg_latest.pt'
        self.best_weights_name = 'seg_best.pt'
        self.latest_weights_file = os.path.join(self.snapshot_path, self.latest_weights_name)
        self.best_weights_file = os.path.join(self.snapshot_path, self.best_weights_name)
        self.max_epochs = 100

        self.base_lr = 0.001
        self.optimizer_config = {0: {'optimizer': 'RMSprop',
                                     'alpha': 0.9,
                                     'eps': 1e-08,
                                     'weight_decay': 0}
                                 }

        self.lr_scheduler_config = {'lr_type': 'CosineLR',
                                    'warm_epoch': -1,
                                    'warmup_iters': 2000}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 2
        self.freeze_layer_name = "base_convActivationBlock_8"

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"


