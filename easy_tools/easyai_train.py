#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from easy_tools.classify_window import ClassifyTrainWindow
from easy_tools.det2d_window import Detection2dTrainWindow
from easy_tools.seg_window import SegmentTrainWindow


class EasyAiTrainWindow(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cls_window = ClassifyTrainWindow()
        self.det2d_window = Detection2dTrainWindow()
        self.seg_window = SegmentTrainWindow()
        self.init_ui()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', '你确认要退出么?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        pass

    def init_ui(self):
        self.addTab(self.cls_window, "ClassNet Train")
        self.addTab(self.det2d_window, "DeNeT Train")
        self.addTab(self.seg_window, "SegNet Train")

        # self.setGeometry(300, 300, 300, 220)
        self.setMinimumSize(QSize(600, 500))
        self.setMaximumSize(QSize(600, 500))
        self.setWindowTitle('easyai train')
        self.setWindowIcon(QIcon('logo.png'))


def main():
    app = QApplication(sys.argv)
    window = EasyAiTrainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

