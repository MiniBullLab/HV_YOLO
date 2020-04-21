#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import TaskArgumentsParse
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.tasks.seg.segment import Segmentation
from easyai.base_name.task_name import TaskName


class InferenceTask():

    def __init__(self, input_path, weight_path):
        self.input_path = input_path
        self.weight_path = weight_path

    def detect2d_task(self, cfg_path, gpu_id, config_path):
        det2d = Detection2d(cfg_path, gpu_id, config_path)
        det2d.load_weights(self.weight_path)
        det2d.process(self.input_path)

    def segment_task(self, cfg_path, gpu_id, config_path):
        seg = Segmentation(cfg_path, gpu_id, config_path)
        seg.load_weights(self.weight_path)
        seg.process(self.input_path)


def main():
    print("process start...")
    options = TaskArgumentsParse.inference_parse_arguments()
    inference_task = InferenceTask(options.inputPath, options.model)
    if options.task_name == TaskName.Detect2d_Task:
        inference_task.detect2d_task(options.model, 0, options.config_path)
    elif options.task_name == TaskName.Segment_Task:
        inference_task.segment_task(options.model, 0, options.config_path)
    print("process end!")


if __name__ == '__main__':
    main()
