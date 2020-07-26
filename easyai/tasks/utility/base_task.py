#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import signal
from easyai.config.utility.config_factory import ConfigFactory


class DelayedKeyboardInterrupt():
    def __init__(self):
        self.signal_received = None

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class BaseTask():

    def __init__(self):
        self.task_name = None
        self.config_factory = ConfigFactory()

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_task_name(self):
        return self.task_name
