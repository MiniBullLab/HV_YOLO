#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.evaluation.calculate_mAp import CalculateMeanAp
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName


class OfflineDet2dEvaluation():

    def __init__(self, class_name):
        self.class_name = class_name
        self.evaluator = CalculateMeanAp(class_name)

    def process(self, test_path, target_path):
        mAP, aps = self.evaluator.result_eval(test_path, target_path)
        return mAP, aps


def main():
    print("start...")
    options = ToolArgumentsParse.test_path_parse()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.Detect2d_Task, config_path=options.config_path)
    test = OfflineDet2dEvaluation(task_config.detect2d_class)
    mAP, _ = test.process(options.inputPath, options.targetPath)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()

