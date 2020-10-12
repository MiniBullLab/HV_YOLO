#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_train_config import ImageTrainConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Segment_Task)
class SegmentionConfig(ImageTrainConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Segment_Task)
        # data
        self.seg_label_type = None
        self.segment_class = None
        self.save_result_dir_name = "segment_results"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_dir_name)
        # test
        # train
        self.log_name = "segment"
        self.train_data_augment = True

        self.config_path = os.path.join(self.config_save_dir, "segmention_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('seg_label_type', None) is not None:
            self.seg_label_type = int(config_dict['seg_label_type'])
        if config_dict.get('segment_class', None) is not None:
            self.segment_class = list(config_dict['segment_class'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['seg_label_type'] = self.seg_label_type
        config_dict['segment_class'] = self.segment_class

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment

    def get_data_default_value(self):
        self.image_size = (500, 400)  # w * H
        self.data_channel = 3
        self.seg_label_type = 1
        self.segment_class = [('background', '255'),
                              ('lane', '0')]

        self.resize_type = 1
        self.normalize_type = 0

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'seg_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.log_name = "segment"
        self.train_data_augment = False
        self.train_batch_size = 1
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'seg_latest.pt'
        self.best_weights_name = 'seg_best.pt'
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.base_lr = 0.001
        self.optimizer_config = {0: {'optimizer': 'RMSprop',
                                     'alpha': 0.9,
                                     'eps': 1e-08,
                                     'weight_decay': 0}
                                 }

        self.lr_scheduler_config = {'type': 'CosineLR',
                                    'warmup_type': 2,
                                    'warmup_iters': 5}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 2
        self.freeze_layer_name = "base_convBNActivationBlock_38"

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"


