#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.task_name import TaskName
from easyai.config.classify_config import ClassifyConfig
from easyai.config.detect2d_config import Detect2dConfig
from easyai.config.segment_config import SegmentionConfig


class ConfigFactory():

    def __init__(self):
        pass

    def get_config(self, task_name, config_path=None):
        task_name = task_name.strip()
        result = None
        if task_name == TaskName.Classify_Task:
            result = ClassifyConfig()
            result.load_config(config_path)
        elif task_name == TaskName.Detect2d_Task:
            result = Detect2dConfig()
            result.load_config(config_path)
        elif task_name == TaskName.Segment_Task:
            result = SegmentionConfig()
            result.load_config(config_path)
        else:
            print("%s task not exits" % task_name)
        return result

    def save(self, task_config):
        pass
        # task_config.save_config()
