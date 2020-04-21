#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import abc


class BaseConfig():

    def __init__(self):
        self.task_name = None
        self.root_save_dir = None
        self.model_save_dir_name = None
        self.snapshot_path = None
        self.config_save_dir_name = None
        self.config_save_dir = None
        self.config_path = None
        self.log_name = "easy"
        self.get_base_default_value()

    @abc.abstractmethod
    def load_config(self, config_path):
        pass

    @abc.abstractmethod
    def save_config(self):
        pass

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_task_name(self):
        return self.task_name

    def get_base_default_value(self):
        self.root_save_dir = "./.log"
        self.model_save_dir_name = "snapshot"
        self.config_save_dir_name = "config"
        self.log_name = "easy"

        self.snapshot_path = os.path.join(self.root_save_dir, self.model_save_dir_name)

        self.config_save_dir = os.path.join(self.root_save_dir, self.config_save_dir_name)
        self.config_path = os.path.join(self.config_save_dir, "config.json")
