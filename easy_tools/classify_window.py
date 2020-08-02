#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class ClassifyTrainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.process = None

    def init_ui(self):
        self.train_data_button = QPushButton('open train dataset')
        self.train_data_button.setCheckable(True)
        self.train_data_button.clicked[bool].connect(self.open_train_dataset)
        self.train_data_txt = QLineEdit()
        self.train_data_txt.setEnabled(False)
        layout1 = QHBoxLayout()
        layout1.setSpacing(20)
        layout1.addWidget(self.train_data_button)
        layout1.addWidget(self.train_data_txt)

        self.val_data_button = QPushButton('open val dataset')
        self.val_data_button.setCheckable(True)
        self.val_data_button.clicked[bool].connect(self.open_val_dataset)
        self.val_data_txt = QLineEdit()
        self.val_data_txt.setEnabled(False)
        layout2 = QHBoxLayout()
        layout2.setSpacing(20)
        layout2.addWidget(self.val_data_button)
        layout2.addWidget(self.val_data_txt)

        self.start_train_button = QPushButton('start train')
        self.start_train_button.setCheckable(True)
        self.start_train_button.setEnabled(False)
        self.start_train_button.clicked[bool].connect(self.start_train)

        self.continue_train_button = QPushButton('continue train')
        self.continue_train_button.setCheckable(True)
        self.continue_train_button.setEnabled(False)
        self.continue_train_button.clicked[bool].connect(self.continue_train)

        self.stop_train_button = QPushButton('start train')
        self.stop_train_button.setCheckable(True)
        self.stop_train_button.setEnabled(False)
        self.stop_train_button.clicked[bool].connect(self.stop_train)

        convert_model_button = QPushButton('model convert')
        convert_model_button.setCheckable(True)
        convert_model_button.setEnabled(False)
        convert_model_button.clicked[bool].connect(self.arm_model_convert)

        layout3 = QHBoxLayout()
        layout3.setSpacing(20)
        layout3.addWidget(self.start_train_button)
        layout3.addWidget(self.continue_train_button)
        layout3.addWidget(self.stop_train_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout1)
        main_layout.addLayout(layout2)
        main_layout.addLayout(layout3)

        self.setLayout(main_layout)

    def open_train_dataset(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "打开train dataset", ".", "txt files(*.txt)")
        if txt_path.strip():
            print("%s error" % txt_path)
            return
        self.train_data_txt.setText(txt_path.strip())
        self.start_train_button.setEnabled(True)
        self.continue_train_button.setEnabled(True)

    def open_val_dataset(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "打开val dataset", ".", "txt files(*.txt)")
        if txt_path.strip():
            print("%s error" % txt_path)
            return
        self.val_data_txt.setText(txt_path.strip())

    def start_train(self, pressed):
        os.system("rm -rf ./log/snapshot/cls_latest.pt")
        os.system("rm -rf ./log/snapshot/cls_best.pt")
        os.system("rm -rf classify_log.txt")
        cmd_str = "./amba_scripts/ClassNET_tool.sh > classify_log.txt"
        env = QProcessEnvironment.systemEnvironment()
        self.process = QProcess()
        self.process.setProcessEnvironment(env)
        self.process.start(cmd_str)
        self.process.finished.connect(self.process_finished)
        self.train_data_button.setEnabled(False)
        self.val_data_button.setEnabled(False)
        self.start_train_button.setEnabled(False)
        self.continue_train_button.setEnabled(False)
        self.stop_train_button.setEnabled(True)

    def continue_train(self, pressed):
        cmd_str = "./amba_scripts/ClassNET_tool.sh > classify_log.txt"
        env = QProcessEnvironment.systemEnvironment()
        self.process = QProcess()
        self.process.setProcessEnvironment(env)
        self.process.start(cmd_str)
        self.process.finished.connect(self.process_finished)
        self.train_data_button.setEnabled(False)
        self.val_data_button.setEnabled(False)

    def stop_train(self, pressed):
        if self.process is not None:
            print("kill pid:", self.process.processId())
            self.process.kill()

    def arm_model_convert(self, pressed):
        pass

    def process_finished(self):
        pass
