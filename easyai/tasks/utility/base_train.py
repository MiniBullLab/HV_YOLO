#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
from easyai.helper.timer_process import TimerProcess


class BaseTrain():

    def __init__(self):
        self.timer = TimerProcess()

    @abc.abstractmethod
    def load_param(self, latest_weights_path):
        pass

    @abc.abstractmethod
    def train(self, train_path, val_path):
        pass

    @abc.abstractmethod
    def compute_loss(self, output_list, targets):
        pass

