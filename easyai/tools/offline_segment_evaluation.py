#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.helper.imageProcess import ImageProcess
from easyai.tools.convert_segment_label import ConvertSegmentionLable
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.evaluation.segmention_metric import SegmentionMetric
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName


class OfflineSegmentEvaluation():

    def __init__(self, seg_label_type=0, segment_class=None):
        self.seg_label_type = seg_label_type
        self.segment_class = segment_class
        self.metric = SegmentionMetric(len(self.segment_class))
        self.image_process = ImageProcess()
        self.label_converter = ConvertSegmentionLable()
        self.segment_sample = SegmentSample(None)

    def process(self, test_dir, target_path):
        self.metric.reset()
        target_data_list = self.segment_sample.get_image_and_label_list(target_path)
        for image_path, label_path in target_data_list:
            path, filename_post = os.path.split(label_path)
            test_path = os.path.join(test_dir, filename_post)
            test_data = self.read_label_image(test_path, 0)
            target_data = self.read_label_image(label_path, self.seg_label_type)
            self.metric.numpy_eval(test_data, target_data)
        score, class_score = self.metric.get_score()
        self.print_evaluation(score)
        return score, class_score

    def read_label_image(self, label_path, label_type):
        if label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.label_converter.process_segment_label(label_path,
                                                              label_type,
                                                              self.segment_class)
        return mask

    def print_evaluation(self, score):
        for k, v in score.items():
            print(k, v)


def main():
    print("start...")
    options = ToolArgumentsParse.test_path_parse()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.Segment_Task, config_path=options.config_path)
    test = OfflineSegmentEvaluation(task_config.seg_label_type,
                                    task_config.segment_class)
    value = test.process(options.inputPath, options.targetPath)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()


