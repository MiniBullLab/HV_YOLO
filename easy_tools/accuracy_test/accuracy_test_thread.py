#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from PyQt5.QtCore import QThread, pyqtSignal
from easyai.tools.offline_evaluation import OfflineEvaluation


class AccuracyTestThread(QThread):

    signal_finish = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flag = -1
        self.is_start = False
        self.task_name = None
        self.config_path = None
        self.target_path = None
        self.arm_result_path = None
        self.model_name = None
        self.weight_path = None
        self.test = None

    def set_param(self, flag, task_name, target_path,
                  arm_result_path=None, config_path=None,
                  model_name=None, weight_path=None):
        self.flag = flag
        self.task_name = task_name
        self.target_path = target_path
        self.arm_result_path = arm_result_path
        self.config_path = config_path
        self.model_name = model_name
        self.weight_path = weight_path
        self.test = OfflineEvaluation(self.task_name, self.target_path,
                                      self.arm_result_path, self.config_path)

    def set_start(self):
        self.is_start = True

    def __del__(self):
        self.is_start = False
        self.wait()

    def run(self):
        while self.is_start:
            if self.flag == 0:
                self.test.pc_test(self.model_name, self.weight_path)
            elif self.flag == 1:
                self.test.arm_test()
            elif self.flag == 2:
                self.test.pc_arm_test(self.model_name, self.weight_path)
            self.is_start = False
        self.signal_finish.emit('finish')

