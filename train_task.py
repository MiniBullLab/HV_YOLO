#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from tasks.det.detect_train import DetectionTrain
from tasks.seg.segmentTrain import SegmentTrain
from helper.arguments_parse import ArgumentsParse


def detect_task(options):
    detect_train = DetectionTrain(options.cfg, 0)
    detect_train.train(options.trainPath, options.valPath)


def segment_task(options):
    segment_train = SegmentTrain(options.cfg, 0)
    segment_train.train(options.trainPath, options.valPath)


def main():
    print("process start...")
    options = ArgumentsParse.train_input_parse()
    if options.task_name == "detect":
        detect_task(options)
    elif options.task_name == "segment":
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
