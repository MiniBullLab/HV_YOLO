#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os

root_save_dir = "./log"
model_save_dir = "snapshot"


class BaseConfig():

    def __init__(self):
        self.root_save_dir = None
        self.model_save_dir = None
        self.config_save_dir = None
        self.snapshot_path = None
        self.log_name = "easy"
        self.get_base_default_value()
        if self.root_save_dir is not None and not os.path.exists(self.root_save_dir):
            os.makedirs(self.root_save_dir, exist_ok=True)

    def get_base_default_value(self):
        self.root_save_dir = "./log"
        self.model_save_dir = "snapshot"
        self.config_save_dir = "config"
        self.log_name = "easy"

        self.snapshot_path = os.path.join(self.root_save_dir, self.model_save_dir)
