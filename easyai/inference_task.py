#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import ArgumentsParse
from easyai.tasks.det2d.detect import Detection
from easyai.tasks.seg.segment import Segmentation


def detect_task(options):
    detect = Detection(options.cfg, 0)
    detect.load_weights(options.weights)
    detect.detect(options.inputPath)


def segment_task(options):
    segment = Segmentation(options.cfg, 0)
    segment.load_weights(options.weights)
    segment.segment(options.inputPath)


def main():
    print("process start...")
    options = ArgumentsParse.parse_arguments()
    if options.task_name == "detect":
        detect_task(options)
    elif options.task_name == "segment":
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
