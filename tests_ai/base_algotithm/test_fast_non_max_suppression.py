#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_fast_non_max_suppression.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import torch
import time
from easyai.base_algorithm.fast_non_max_suppression import FastNonMaxSuppression


def test():
    fast_nms = FastNonMaxSuppression()

    random_boxes = torch.rand([10000, 5]).numpy()
    time_1 = time.time()
    result = fast_nms.nms(random_boxes, 0.05)
    print("time cost {}".format(time.time() - time_1))


if __name__ == "__main__":
    test()