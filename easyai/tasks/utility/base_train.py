#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
from easyai.helper.timer_process import TimerProcess


class BaseTrain():

    def __init__(self):
        self.timer = TimerProcess()

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

