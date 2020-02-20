#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import ArgumentsParse
from easyai.tasks.cls.classify_test import ClassifyTest
from easyai.tasks.det2d.detect2d_test import Detection2dTest
from easyai.tasks.seg.segment_test import SegmentionTest
from easyai.base_name.task_name import TaskName


def classify_task(options):
    test = ClassifyTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)


def detect_task(options):
    detect_test = Detection2dTest(options.cfg, 0)
    detect_test.load_weights(options.weights)
    detect_test.test(options.valPath)


def segment_task(options):
    test = SegmentionTest(options.cfg, 0)
    test.load_weights(options.weights)
    test.test(options.valPath)


def main():
    print("process start...")
    options = ArgumentsParse.test_input_parse()
    if options.task_name == TaskName.Classify_Task:
        classify_task(options)
    elif options.task_name == TaskName.Detect2d_Task:
        detect_task(options)
    elif options.task_name == TaskName.Segment_Task:
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
