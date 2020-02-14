#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
from easyai.helper.timer_process import TimerProcess


class BaseTest():

    def __init__(self):
        self.timer = TimerProcess()

    @abc.abstractmethod
    def load_weights(self, weights_path):
        pass

    @abc.abstractmethod
    def test(self, val_path):
        pass
