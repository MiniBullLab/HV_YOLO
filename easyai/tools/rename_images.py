#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from easyai.helper.dirProcess import DirProcess


def rename_imaegs(input_dir):
    dir_process = DirProcess()
    for image_path in dir_process.getDirFiles(input_dir, "*.png"):
        print(image_path)
        path, file_name_and_post = os.path.split(image_path)
        image_name, post = os.path.splitext(file_name_and_post)
        new_name = image_name.split("_")[0] + post
        new_path = os.path.join(path, new_name)
        os.rename(image_path, new_path)


def test():
    print("start...")
    rename_imaegs("/home/lpj/github/data/LED/dataset/defect_N/test_label")
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()
