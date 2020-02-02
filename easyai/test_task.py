#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import ArgumentsParse
from easyai.tasks.cls.classify_test import ClassifyTest
from easyai.tasks.det2d.detect_test import DetectionTest
from easyai.tasks.seg.segment_test import SegmentionTest


def classify_task(options):
    test = ClassifyTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)


def detect_task(options):
    detect_test = DetectionTest(options.cfg, 0)
    detect_test.load_weights(options.weights)
    detect_test.test(options.valPath)


def segment_task(options):
    test = SegmentionTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)


def main():
    print("process start...")
    options = ArgumentsParse.test_input_parse()
    if options.task_name == "classify":
        classify_task(options)
    elif options.task_name == "detect":
        detect_task(options)
    elif options.task_name == "segment":
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
