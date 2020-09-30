#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import inspect
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from easy_tools.train_gui.process_status import ProcessStatus


class SegmentTrainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.process = None
        self.dir_path = "."
        self.is_status = ProcessStatus.UNKNOW
        self.write_file = None
        self.save_log = "segmentation_log.txt"
        current_path = inspect.getfile(inspect.currentframe())
        dir_name = os.path.dirname(current_path)
        self.cmd_str = os.path.join(dir_name, "../train_scripts/SegNET_tool.sh")

    def closeEvent(self, event):
        if self.is_status == ProcessStatus.UNKNOW:
            pass
        elif(self.is_status == ProcessStatus.START_TRAIN or
             self.is_status == ProcessStatus.CONTINUE_TRAIN):
            reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                event.accept()
                self.kill_process()
                self.write_log_text()
            else:
                event.ignore()
        elif self.is_status != ProcessStatus.THREAD_STOP:
            self.write_log_text()

    def open_train_dataset(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "open train dataset", self.dir_path, "txt files(*.txt)")
        if txt_path.strip():
            self.train_data_txt.setText(txt_path.strip())
            self.start_train_button.setEnabled(True)
            self.continue_train_button.setEnabled(True)
            self.dir_path, _ = os.path.split(txt_path)
        else:
            print("%s error" % txt_path)
            return

    def open_val_dataset(self, pressed):
        txt_path, _ = QFileDialog.getOpenFileName(self, "open val dataset", self.dir_path, "txt files(*.txt)")
        if txt_path.strip():
            self.val_data_txt.setText(txt_path.strip())
            self.dir_path, _ = os.path.split(txt_path)
        else:
            print("%s error" % txt_path)
            return

    def start_train(self, pressed):
        os.system("rm -rf %s/snapshot/seg_latest.pt" % ProcessStatus.LOG_ROOT)
        os.system("rm -rf %s/snapshot/seg_best.pt" % ProcessStatus.LOG_ROOT)
        os.system("rm -rf %s" % self.save_log)
        arguments = [self.train_data_txt.text(), self.val_data_txt.text()]
        env = QProcessEnvironment.systemEnvironment()
        self.process = QProcess()
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        # self.process.setStandardOutputFile(self.save_log, QIODevice.WriteOnly)
        # self.process.setStandardErrorFile(self.save_log, QIODevice.WriteOnly)
        self.process.readyReadStandardOutput.connect(self.process_standard_output)
        self.process.readyReadStandardError.connect(self.process_standard_error)
        self.process.finished.connect(self.process_finished)
        self.process.start(self.cmd_str, arguments)
        self.process.waitForStarted()
        # self.process.waitForFinished(-1)
        self.train_data_button.setEnabled(False)
        self.val_data_button.setEnabled(False)
        self.start_train_button.setEnabled(False)
        self.continue_train_button.setEnabled(False)
        self.stop_train_button.setEnabled(True)
        self.is_status = ProcessStatus.START_TRAIN
        self.text_browser.clear()
        self.text_browser.append("segmentation start train!")
        try:
            self.write_file = open(self.save_log, 'w')
        except Exception as e:
            print(e)

    def continue_train(self, pressed):
        arguments = [self.train_data_txt.text(), self.val_data_txt.text()]
        env = QProcessEnvironment.systemEnvironment()
        self.process = QProcess()
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.process_standard_output)
        self.process.readyReadStandardError.connect(self.process_standard_error)
        self.process.finished.connect(self.process_finished)
        self.process.start(self.cmd_str, arguments)
        self.process.waitForStarted()
        # self.process.waitForFinished(-1)
        self.train_data_button.setEnabled(False)
        self.val_data_button.setEnabled(False)
        self.start_train_button.setEnabled(False)
        self.continue_train_button.setEnabled(False)
        self.stop_train_button.setEnabled(True)
        self.is_status = ProcessStatus.CONTINUE_TRAIN
        self.text_browser.append("segmentation continue train!")
        try:
            self.write_file = open(self.save_log, 'a')
        except Exception as e:
            print(e)

    def stop_train(self, pressed):
        self.kill_process()
        self.is_status = ProcessStatus.STOP_TRAIN
        self.train_data_button.setEnabled(True)
        self.val_data_button.setEnabled(True)
        self.start_train_button.setEnabled(True)
        self.continue_train_button.setEnabled(True)
        self.stop_train_button.setEnabled(False)
        self.text_browser.append("segmentation stop train!")

    def arm_model_convert(self, pressed):
        pass

    def process_finished(self):
        if self.is_status != ProcessStatus.CONTINUE_TRAIN:
            self.train_data_button.setEnabled(True)
            self.val_data_button.setEnabled(True)
            self.start_train_button.setEnabled(True)
            self.continue_train_button.setEnabled(False)
            self.stop_train_button.setEnabled(False)
            self.text_browser.append("segmentation train end!")
        self.process.close()
        self.write_log_text()
        self.is_status = ProcessStatus.THREAD_STOP

    def process_standard_output(self):
        temp_str = self.process.readAllStandardOutput().data().decode('utf-8')
        self.text_browser.append(temp_str)

    def process_standard_error(self):
        temp_str = self.process.readAllStandardError().data().decode('utf-8')
        self.text_browser.append(temp_str)

    def kill_process(self):
        if self.process is not None:
            self.text_browser.append("seg kill pid: %d" % self.process.processId())
            self.process.kill()
            self.process.waitForFinished(-1)

    def write_log_text(self):
        try:
            if self.write_file is not None:
                str_text = self.text_browser.toPlainText()
                temp = str(str_text)
                self.write_file.write('{}'.format(temp))
                self.write_file.close()
        except Exception as e:
            print(e)

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

        self.stop_train_button = QPushButton('stop train')
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

        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)
        top_layout.addLayout(layout1)
        top_layout.addLayout(layout2)
        top_layout.addLayout(layout3)

        self.text_browser = QTextBrowser()
        self.text_browser.setReadOnly(True)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.addLayout(top_layout, 1)
        main_layout.addWidget(self.text_browser, 3)

        self.setLayout(main_layout)
