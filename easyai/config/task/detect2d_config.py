#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import codecs
import json
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig


class Detect2dConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Detect2d_Task)
        # data
        self.class_name = None
        self.confidence_th = 1.0
        self.nms_th = 1.0
        # test
        self.save_result_dir = os.path.join(self.root_save_dir, 'det2d_results')
        self.save_evaluation_path = os.path.join(self.root_save_dir, 'det2d_evaluation.txt')
        # train
        self.log_name = "detect2d"
        self.train_data_augment = True
        self.train_multi_scale = False
        self.balanced_sample = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'det2d_latest.pt'
        self.best_weights_name = 'det2d_best.pt'
        self.latest_weights_file = None
        self.best_weights_file = None
        self.accumulated_batches = 1
        self.display = 1

        self.freeze_layer_type = 0
        self.freeze_layer_name = None

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = None

        self.config_path = os.path.join(self.config_save_dir, "detection2d_config.json")

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
        if config_dict.get('class_name', None) is not None:
            self.class_name = tuple(config_dict['class_name'])
        if config_dict.get('confidence_th', None) is not None:
            self.confidence_th = float(config_dict['confidence_th'])
        if config_dict.get('nms_th', None) is not None:
            self.nms_th = float(config_dict['nms_th'])

    def save_data_value(self, config_dict):
        config_dict['image_size'] = self.image_size
        config_dict['class_name'] = self.class_name
        config_dict['confidence_th'] = self.confidence_th
        config_dict['nms_th'] = self.nms_th

    def load_test_value(self, config_dict):
        if config_dict.get('test_batch_size', None) is not None:
            self.test_batch_size = int(config_dict['test_batch_size'])

    def save_test_value(self, config_dict):
        config_dict['test_batch_size'] = self.test_batch_size

    def load_train_value(self, config_dict):
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])
        if config_dict.get('train_multi_scale', None) is not None:
            self.train_multi_scale = bool(config_dict['train_multi_scale'])
        if config_dict.get('balanced_sample', None) is not None:
            self.balanced_sample = bool(config_dict['balanced_sample'])
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
        config_dict['train_multi_scale'] = self.train_multi_scale
        config_dict['balanced_sample'] = self.balanced_sample
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
        self.image_size = (640, 352)  # W * H
        self.class_name = ('bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck')
        self.confidence_th = 0.5
        self.nms_th = 0.45

    def get_test_default_value(self):
        self.test_batch_size = 1

    def get_train_default_value(self):
        self.log_name = "detect2d"
        self.train_data_augment = True
        self.train_multi_scale = False
        self.balanced_sample = False
        self.train_batch_size = 16
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'det2d_latest.pt'
        self.best_weights_name = 'det2d_best.pt'
        self.latest_weights_file = os.path.join(self.snapshot_path, self.latest_weights_name)
        self.best_weights_file = os.path.join(self.snapshot_path, self.best_weights_name)

        self.max_epochs = 100

        self.base_lr = 2e-4
        self.optimizer_config = {0: {'optimizer': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'lr_type': 'MultiStageLR',
                                    'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
                                    'warm_epoch': 0,
                                    'warmup_iters': 1000}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 0
        self.freeze_layer_name = "route_0"

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"

