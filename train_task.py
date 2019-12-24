#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from tasks.cls.classify_train import ClassifyTrain
from tasks.det.detect_train import DetectionTrain
from tasks.seg.segment_train import SegmentionTrain
from helper.arguments_parse import ArgumentsParse


def classify_task(options):
    classify_train = ClassifyTrain(options.cfg, 0)
    classify_train.train(options.trainPath, options.valPath)


def detect_task(options):
    detect_train = DetectionTrain(options.cfg, 0)
    detect_train.train(options.trainPath, options.valPath)


def segment_task(options):
    segment_train = SegmentionTrain(options.cfg, 0)
    segment_train.train(options.trainPath, options.valPath)


def main():
    print("process start...")
    options = ArgumentsParse.train_input_parse()
    if options.task_name == "classify":
        classify_task(options)
    elif options.task_name == "detect":
        detect_task(options)
    elif options.task_name == "segment":
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
