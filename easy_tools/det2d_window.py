#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from PyQt5.QtWidgets import *


class Detection2dTrainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        train_data_button = QPushButton('open train dataset')
        train_data_button.setCheckable(True)
        train_data_button.clicked[bool].connect(self.open_train_dataset)
        self.train_data_txt = QLineEdit()
        self.train_data_txt.setEnabled(False)
        layout1 = QHBoxLayout()
        layout1.setSpacing(20)
        layout1.addWidget(train_data_button)
        layout1.addWidget(self.train_data_txt)

        val_data_button = QPushButton('open val dataset')
        val_data_button.setCheckable(True)
        val_data_button.clicked[bool].connect(self.open_val_dataset)
        self.val_data_txt = QLineEdit()
        self.val_data_txt.setEnabled(False)
        layout2 = QHBoxLayout()
        layout2.setSpacing(20)
        layout2.addWidget(val_data_button)
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
        pass

    def continue_train(self, pressed):
        pass

    def stop_train(self, pressed):
        pass

