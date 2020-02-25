#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import ArgumentsParse
from easyai.tasks.cls.classify_train import ClassifyTrain
from easyai.tasks.det2d.detect2d_train import Detection2dTrain
from easyai.tasks.seg.segment_train import SegmentionTrain
from easyai.base_name.task_name import TaskName


def classify_task(options):
    classify_train = ClassifyTrain(options.cfg, 0)
    classify_train.load_pretrain_model(options.pretrainModel)
    classify_train.train(options.trainPath, options.valPath)


def detect_task(options):
    detect_train = Detection2dTrain(options.cfg, 0)
    detect_train.load_pretrain_model(options.pretrainModel)
    detect_train.train(options.trainPath, options.valPath)


def segment_task(options):
    segment_train = SegmentionTrain(options.cfg, 0)
    segment_train.load_pretrain_model(options.pretrainModel)
    segment_train.train(options.trainPath, options.valPath)


def main():
    print("process start...")
    options = ArgumentsParse.train_input_parse()
    if options.task_name == TaskName.Classify_Task:
        classify_task(options)
    elif options.task_name == TaskName.Detect2d_Task:
        detect_task(options)
    elif options.task_name == TaskName.Segment_Task:
        segment_task(options)
    print("process end!")


if __name__ == '__main__':
    main()
