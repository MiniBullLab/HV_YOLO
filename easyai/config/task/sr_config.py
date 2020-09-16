#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_train_config import ImageTrainConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.SuperResolution_Task)
class SuperResolutionConfig(ImageTrainConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.SuperResolution_Task)
        # data
        self.upscale_factor = 1
        # test
        # train
        self.log_name = TaskName.SuperResolution_Task
        self.train_data_augment = False

        self.config_path = os.path.join(self.config_save_dir, "super_resolution_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('upscale_factor', None) is not None:
            self.upscale_factor = int(config_dict['upscale_factor'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['upscale_factor'] = self.upscale_factor

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment

    def get_data_default_value(self):
        self.image_size = (720, 1024)  # w * H
        self.data_channel = 1
        self.upscale_factor = 2

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'sr_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data_augment = False
        self.train_batch_size = 2
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'sr_latest.pt'
        self.best_weights_name = 'sr_best.pt'
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.base_lr = 1e-3
        self.optimizer_config = {0: {'optimizer': 'Adam',
                                     'betas': (0.9, 0.999),
                                     'eps': 1e-08,
                                     'weight_decay': 0}
                                 }

        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
                                    'warmup_type': 0,
                                    'warmup_iters': 1000}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 0
        self.freeze_layer_name = "route_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"
