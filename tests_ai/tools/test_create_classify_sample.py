#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_create_classify_sample.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.sample.create_classify_sample import CreateClassifySample

def test(input_dir, output_dir, flag, probability):
    print("start...")
    test = CreateClassifySample()
    test.process_sample(input_dir,
                        output_dir,
                        flag,
                        probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    input_dir = "/home/wfw/data/VOCdevkit/Tree_Ring_classify/JPEGImages"
    output_dir = "/home/wfw/data/VOCdevkit/Tree_Ring_classify/ImageSets"
    flag = "train_val"
    probability = 10
    test(input_dir, output_dir, flag, probability=probability)