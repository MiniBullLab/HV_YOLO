#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from PyQt5.QtCore import *


class ProcessThread(QThread):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.cmd_str = None
        self.is_start = False
        self.process = None

    def init_cmd(self, cmd):
        self.cmd_str = cmd

    def start_thread(self):
        self.is_start = True

    def stop_thread(self):
        if self.process is not None:
            print("kill pid:", self.process.processId())
            self.process.kill()
        self.is_start = False

    def process_finish(self):
        pass

    def run(self):
        while self.is_start:
            if self.cmd_str is not None:
                env = QProcessEnvironment.systemEnvironment()
                self.process = QProcess()
                self.process.setProcessEnvironment(env)
                self.process.start(self.cmd_str)

    def __del__(self):
        self.stop_thread()
        self.wait()
