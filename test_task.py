#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from tasks.det.detect_test import DetectionTest
from tasks.seg.segmentTest import SegmentTest
from helper.arguments_parse import ArgumentsParse


def detect_task(options):
    detect_test = DetectionTest(options.cfg, 0)
    detect_test.load_weights(options.weights)
    detect_test.test(options.valPath)


def segment_task(options):
    test = SegmentTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)


def main():
    print("process start...")
    options = ArgumentsParse.test_input_parse()
    if options.task_name == "detect":
        detect_task(options)
    elif options.task_name == "segment":
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
