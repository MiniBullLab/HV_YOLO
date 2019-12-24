#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from tasks.cls.classify_test import ClassifyTest
from tasks.det.detect_test import DetectionTest
from tasks.seg.segment_test import SegmentionTest
from helper.arguments_parse import ArgumentsParse


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
