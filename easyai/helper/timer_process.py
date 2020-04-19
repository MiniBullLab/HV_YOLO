#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import time


class TimerProcess():

    def __init__(self):
        self.start_time = 0
        self.tic()

    def tic(self):
        self.start_time = time.time()

    def toc(self, reset=False):
        now_time = time.time()
        diff_time = now_time - self.start_time
        if reset:
            self.tic()
        return diff_time
