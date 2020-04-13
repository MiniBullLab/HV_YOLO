#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import ArgumentsParse
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.tasks.seg.segment import Segmentation
from easyai.base_name.task_name import TaskName


def detect_task(options):
    detect = Detection2d(options.cfg, 0)
    detect.load_weights(options.weights)
    detect.process(options.inputPath)


def segment_task(options):
    segment = Segmentation(options.cfg, 0)
    segment.load_weights(options.weights)
    segment.process(options.inputPath)


def main():
    print("process start...")
    options = ArgumentsParse.parse_arguments()
    if options.task_name == TaskName.Detect2d_Task:
        detect_task(options)
    elif options.task_name == TaskName.Segment_Task:
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
