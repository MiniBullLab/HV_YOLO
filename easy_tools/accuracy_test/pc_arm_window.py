#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from easyai.base_name.task_name import TaskName
from easy_tools.accuracy_test.accuracy_test_thread import AccuracyTestThread
import inspect


class PCAndArmTestWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.test_process = AccuracyTestThread()
        self.test_process.signal_finish[str].connect(self.process_finished)
        self.dir_path = "."

    def open_val_dataset(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "open val dataset", self.dir_path, "txt files(*.txt)")
        if txt_path.strip():
            self.val_data_txt.setText(txt_path.strip())
            self.start_button.setEnabled(True)
            self.dir_path, _ = os.path.split(txt_path)
        else:
            print("%s error" % txt_path)
            return

    def open_arm_dataset(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "open arm result", self.dir_path, "txt files(*.txt)")
        if txt_path.strip():
            self.arm_data_txt.setText(txt_path.strip())
            self.dir_path, _ = os.path.split(txt_path)
        else:
            print("%s error" % txt_path)
            return

    def open_weight_path(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "open weight file", self.dir_path, "pt files(*.pt)")
        if txt_path.strip():
            self.weight_txt.setText(txt_path.strip())
            self.dir_path, _ = os.path.split(txt_path)
        else:
            print("%s error" % txt_path)
            return

    def open_config_path(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "open config file", self.dir_path, "json files(*.json)")
        if txt_path.strip():
            self.config_txt.setText(txt_path.strip())
            self.dir_path, _ = os.path.split(txt_path)
        else:
            print("%s error" % txt_path)
            return

    def start(self, pressed):
        self.text_browser.clear()
        flag = self.task_type_box.currentIndex()
        task_name = self.task_name_box.currentText()
        target_path = self.val_data_txt.text()
        model_name = None
        current_path = inspect.getfile(inspect.currentframe())
        dir_name = os.path.dirname(current_path)
        if flag != 1:
            if task_name == TaskName.Classify_Task:
                model_name = 'classnet'
            elif task_name == TaskName.Detect2d_Task:
                model_name = 'detnet'
            elif task_name == TaskName.Segment_Task:
                cfg_path = os.path.join(dir_name, "../data/segnet.cfg")
                model_name = cfg_path
        if flag == 0:
            if self.weight_txt.text():
                self.test_process.set_param(flag=flag, task_name=task_name,
                                            target_path=target_path,
                                            model_name=model_name,
                                            weight_path=self.weight_txt.text(),
                                            config_path=self.config_txt.text())
                self.test_process.set_start()
                self.test_process.start()
                self.start_button.setEnabled(False)
            else:
                QMessageBox.information(self, " PC accuracy test",
                                        "parameter error", QMessageBox.Yes)
        elif flag == 1:
            if self.arm_data_txt.text():
                self.test_process.set_param(flag=flag, task_name=task_name,
                                            target_path=target_path,
                                            arm_result_path=self.arm_data_txt.text(),
                                            config_path=self.config_txt.text())
                self.test_process.set_start()
                self.test_process.start()
                self.start_button.setEnabled(False)
            else:
                QMessageBox.information(self, "ARM accuracy test",
                                        "parameter error", QMessageBox.Yes)
        elif flag == 0:
            if self.arm_data_txt.text() and self.weight_txt.text():
                self.test_process.set_param(flag=flag, task_name=task_name,
                                            target_path=target_path,
                                            arm_result_path=self.arm_data_txt.text(),
                                            model_name=model_name,
                                            weight_path=self.weight_txt.text(),
                                            config_path=self.config_txt.text())
                self.test_process.set_start()
                self.test_process.start()
                self.start_button.setEnabled(False)
            else:
                QMessageBox.information(self, "PC and ARM accuracy test",
                                        "parameter error", QMessageBox.Yes)

    def process_finished(self, str_value):
        self.start_button.setEnabled(True)
        self.text_browser.append(str_value)

    def init_ui(self):
        self.task_type_label = QLabel("test type:")
        self.task_type_box = QComboBox()
        self.task_type_box.addItem('PC')
        self.task_type_box.addItem('ARM')
        self.task_type_box.addItem('PC And ARM')

        self.task_name_label = QLabel("task name:")
        self.task_name_box = QComboBox()
        self.task_name_box.addItem(TaskName.Classify_Task)
        self.task_name_box.addItem(TaskName.Detect2d_Task)
        self.task_name_box.addItem(TaskName.Segment_Task)

        layout1 = QHBoxLayout()
        layout1.setSpacing(20)
        layout1.addWidget(self.task_type_label)
        layout1.addWidget(self.task_type_box)
        layout1.addWidget(self.task_name_label)
        layout1.addWidget(self.task_name_box)

        self.val_data_button = QPushButton('open val dataset')
        self.val_data_button.setCheckable(True)
        self.val_data_button.clicked[bool].connect(self.open_val_dataset)
        self.val_data_txt = QLineEdit()
        self.val_data_txt.setEnabled(False)

        layout2 = QHBoxLayout()
        layout2.setSpacing(20)
        layout2.addWidget(self.val_data_button)
        layout2.addWidget(self.val_data_txt)

        self.arm_data_button = QPushButton('open arm result')
        self.arm_data_button.setCheckable(True)
        self.arm_data_button.clicked[bool].connect(self.open_arm_dataset)
        self.arm_data_txt = QLineEdit()
        self.arm_data_txt.setEnabled(False)

        layout3 = QHBoxLayout()
        layout3.setSpacing(20)
        layout3.addWidget(self.arm_data_button)
        layout3.addWidget(self.arm_data_txt)

        self.weight_button = QPushButton('open weight file')
        self.weight_button.setCheckable(True)
        self.weight_button.clicked[bool].connect(self.open_weight_path)
        self.weight_txt = QLineEdit()
        self.weight_txt.setEnabled(False)

        layout4 = QHBoxLayout()
        layout4.setSpacing(20)
        layout4.addWidget(self.weight_button)
        layout4.addWidget(self.weight_txt)

        self.config_button = QPushButton('open config file')
        self.config_button.setCheckable(True)
        self.config_button.clicked[bool].connect(self.open_config_path)
        self.config_txt = QLineEdit()
        self.config_txt.setEnabled(False)

        layout5 = QHBoxLayout()
        layout5.setSpacing(20)
        layout5.addWidget(self.config_button)
        layout5.addWidget(self.config_txt)

        self.start_button = QPushButton('start')
        self.start_button.setCheckable(True)
        self.start_button.setEnabled(False)
        self.start_button.clicked[bool].connect(self.start)

        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)
        top_layout.addLayout(layout1)
        top_layout.addLayout(layout2)
        top_layout.addLayout(layout3)
        top_layout.addLayout(layout4)
        top_layout.addLayout(layout5)
        top_layout.addWidget(self.start_button)

        self.text_browser = QTextBrowser()
        self.text_browser.setReadOnly(True)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.addLayout(top_layout, 4)
        main_layout.addWidget(self.text_browser, 1)

        self.setLayout(main_layout)

        self.setMinimumSize(QSize(600, 500))
        # self.setMaximumSize(QSize(600, 500))
        self.setWindowTitle('easyai train')
        self.setWindowIcon(QIcon('./logo.png'))
