#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import sys
from PyQt5.QtWidgets import QApplication
from easy_tools.accuracy_test.pc_arm_window import PCAndArmTestWindow


def main():
    app = QApplication(sys.argv)
    window = PCAndArmTestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()