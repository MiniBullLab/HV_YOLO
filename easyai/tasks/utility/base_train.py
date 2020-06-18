#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
from easyai.helper.timer_process import TimerProcess
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.torch_utility.torch_freeze_bn import TorchFreezeNormalization
from easyai.utility.train_log import TrainLogger
from easyai.tasks.utility.base_task import BaseTask


class BaseTrain(BaseTask):

    def __init__(self, config_path, task_name):
        super().__init__()
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.freeze_normalization = TorchFreezeNormalization()
        self.model = None
        self.optimizer = None
        self.config_path = config_path
        self.is_sparse = False
        self.sparse_ratio = 0.0
        self.train_task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.train_logger = TrainLogger(self.train_task_config.log_name,
                                        self.train_task_config.root_save_dir)

    @abc.abstractmethod
    def load_latest_param(self, latest_weights_path):
        pass

    @abc.abstractmethod
    def train(self, train_path, val_path):
        pass

    @abc.abstractmethod
    def compute_backward(self, input_datas, targets, setp_index):
        pass

    @abc.abstractmethod
    def compute_loss(self, output_list, targets):
        pass

    def load_pretrain_model(self, weights_path):
        self.torchModelProcess.loadPretainModel(weights_path, self.model)

    def set_is_sparse_train(self, is_sparse=False, sparse_ratio=0.0):
        self.is_sparse = is_sparse
        self.sparse_ratio = sparse_ratio

