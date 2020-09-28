#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_create_segment_sample.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.sample.create_segment_sample import CreateSegmentionSample


def test(input_dir, output_path, probability):
    print("start...")
    test = CreateSegmentionSample()
    test.create_train_and_test(input_dir,
                               output_path,
                               probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    input_dir = "/home/wfw/data/VOCdevkit/LedScratch_segment/JPEGImages"
    output_path = "/home/wfw/data/VOCdevkit/LedScratch_segment/ImageSets"

    # test_balance_sample(config_path, train_path, output_dir)
    test(input_dir, output_path, 10)
