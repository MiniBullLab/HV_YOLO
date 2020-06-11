#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
from easyai.helper.timer_process import TimerProcess
from easyai.tasks.utility.base_task import BaseTask


class BaseTrain(BaseTask):

    def __init__(self, config_path):
        super().__init__()
        self.timer = TimerProcess()
        self.config_path = config_path
        self.is_sparse = False
        self.sparse_ratio = 0.0

    @abc.abstractmethod
    def load_pretrain_model(self, weights_path):
        pass

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

    def set_is_sparse_train(self, is_sparse=False, sparse_ratio=0.0):
        self.is_sparse = is_sparse
        self.sparse_ratio = sparse_ratio

