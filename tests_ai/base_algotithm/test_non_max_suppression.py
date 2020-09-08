#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_non_max_suppression.py
# Author: wfw

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import torch
import time
import random
from easyai.helper.dataType import Rect2D
from easyai.base_algorithm.non_max_suppression import NonMaxSuppression


def test():
    nms = NonMaxSuppression()

    num_objs = 10000

    random_boxes = []
    for i in range(0, num_objs):
        b = Rect2D()
        b.min_corner.x = random.random()
        b.min_corner.y = random.random()
        b.max_corner.x = random.random()
        b.max_corner.y = random.random()
        random_boxes.append(b)

    time_1 = time.time()
    result = nms.nms(random_boxes, 0.05)
    print("time cost {}".format(time.time() - time_1))


if __name__ == "__main__":
    test()