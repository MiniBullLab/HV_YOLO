#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_train_config import ImageTrainConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Detect2d_Task)
class Detect2dConfig(ImageTrainConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Detect2d_Task)
        # data
        self.detect2d_class = None
        self.confidence_th = 1.0
        self.nms_th = 1.0
        self.post_prcoess_type = 0
        # test
        self.save_result_dir = os.path.join(self.root_save_dir, 'det2d_results')
        # train
        self.log_name = "detect2d"
        self.train_data_augment = True
        self.train_multi_scale = False
        self.balanced_sample = False

        self.config_path = os.path.join(self.config_save_dir, "detection2d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('detect2d_class', None) is not None:
            self.detect2d_class = tuple(config_dict['detect2d_class'])
        if config_dict.get('confidence_th', None) is not None:
            self.confidence_th = float(config_dict['confidence_th'])
        if config_dict.get('nms_th', None) is not None:
            self.nms_th = float(config_dict['nms_th'])
        if config_dict.get('post_prcoess_type', None) is not None:
            self.post_prcoess_type = int(config_dict['post_prcoess_type'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['detect2d_class'] = self.detect2d_class
        config_dict['confidence_th'] = self.confidence_th
        config_dict['nms_th'] = self.nms_th
        config_dict['post_prcoess_type'] = self.post_prcoess_type

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])
        if config_dict.get('train_multi_scale', None) is not None:
            self.train_multi_scale = bool(config_dict['train_multi_scale'])
        if config_dict.get('balanced_sample', None) is not None:
            self.balanced_sample = bool(config_dict['balanced_sample'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment
        config_dict['train_multi_scale'] = self.train_multi_scale
        config_dict['balanced_sample'] = self.balanced_sample

    def get_data_default_value(self):
        self.image_size = (416, 416)  # W * H
        self.data_channel = 3
        self.detect2d_class = ("bike",
                               "bus",
                               "car",
                               "motor",
                               "person",
                               "rider",
                               "traffic light",
                               "traffic sign",
                               "train",
                               "truck")
        self.confidence_th = 0.24
        self.nms_th = 0.45
        self.post_prcoess_type = 0

        self.resize_type = 1
        self.normalize_type = 0

        self.save_result_name = "det2d_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'det2d_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.log_name = "detect2d"
        self.train_data_augment = True
        self.train_multi_scale = False
        self.balanced_sample = False
        self.train_batch_size = 1
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'det2d_latest.pt'
        self.best_weights_name = 'det2d_best.pt'
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 100

        self.base_lr = 2e-4
        self.optimizer_config = {0: {'optimizer': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
                                    'warmup_type': 2,
                                    'warmup_iters': 5}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 1
        self.freeze_layer_name = "baseNet_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"

