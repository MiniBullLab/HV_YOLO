#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.evaluation.calculate_mAp import CalculateMeanAp
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.tools.offline_test.base_offline_evaluation import BaseOfflineEvaluation
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName
from easyai.tools.utility.registry import REGISTERED_OFFLINE_EVALUATION


@REGISTERED_OFFLINE_EVALUATION.register_module(TaskName.Detect2d_Task)
class OfflineDet2dEvaluation(BaseOfflineEvaluation):

    def __init__(self, detect2d_class):
        super().__init__()
        self.detect2d_class = detect2d_class
        self.evaluator = CalculateMeanAp(detect2d_class)

    def process(self, test_path, target_path):
        mAP, aps = self.evaluator.result_eval(test_path, target_path)
        # self.evaluator.print_evaluation(aps)
        return mAP, aps

    def print_evaluation(self, value):
        print_str = "Mean AP = {:.4f}\n".format(value[0])
        print_str += "Results:\n"
        for i, ap in enumerate(value[1]):
            temp_str = self.detect2d_class[i] + ': ' + '{:.3f}'.format(ap) + '\n'
            print_str += temp_str
        print(print_str)
        return print_str


def main():
    print("start...")
    options = ToolArgumentsParse.test_path_parse()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.Detect2d_Task, config_path=options.config_path)
    test = OfflineDet2dEvaluation(task_config.detect2d_class)
    value = test.process(options.inputPath, options.targetPath)
    test.print_evaluation(value)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()

