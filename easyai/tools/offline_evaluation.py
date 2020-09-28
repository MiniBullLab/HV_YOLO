#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tools.utility.registry import REGISTERED_OFFLINE_EVALUATION
from easyai.utility.registry import build_from_cfg
from easyai.base_name.task_name import TaskName
from easyai.inference_task import InferenceTask
from easyai.config.utility.config_factory import ConfigFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class OfflineEvaluation():

    def __init__(self, task_name, target_path, arm_result_path=None, config_path=None):
        self.task_name = task_name
        self.config_path = config_path
        self.target_path = target_path
        self.arm_result_path = arm_result_path
        config_factory = ConfigFactory()
        task_config = config_factory.get_config(task_name, config_path)
        self.offline_test = None
        if self.task_name is not None and \
                REGISTERED_OFFLINE_EVALUATION.has_class(self.task_name):
            if self.task_name.strip() == TaskName.Classify_Task:
                test_args = {'type': self.task_name}
                self.offline_test = build_from_cfg(test_args, REGISTERED_OFFLINE_EVALUATION)
            elif self.task_name.strip() == TaskName.Detect2d_Task:
                test_args = {'type': self.task_name,
                             'detect2d_class': task_config.detect2d_class}
                self.offline_test = build_from_cfg(test_args, REGISTERED_OFFLINE_EVALUATION)
            elif self.task_name.strip() == TaskName.Segment_Task:
                test_args = {'type': self.task_name,
                             'seg_label_type': task_config.seg_label_type,
                             'segment_class': task_config.segment_class}
                self.offline_test = build_from_cfg(test_args, REGISTERED_OFFLINE_EVALUATION)
        else:
            print("This task(%s) not exits!" % self.task_name)

    def pc_arm_test(self, model_name, weight_path):
        self.pc_test(model_name, weight_path)
        self.arm_test()

    def pc_test(self, model_name, weight_path):
        inference_task = InferenceTask(self.task_name, self.target_path, False)
        inference_task.infer(model_name, 0, weight_path, self.config_path)
        value = self.offline_test.process(inference_task.task_config.save_result_path,
                                          self.target_path)
        print("pc")
        self.offline_test.print_evaluation(value)
        return value

    def arm_test(self):
        if self.arm_result_path is None:
            return None
        value = self.offline_test.process(self.arm_result_path,
                                          self.target_path)
        print("arm")
        self.offline_test.print_evaluation(value)
        return value


def main():
    print("start...")
    options = ToolArgumentsParse.offline_test_path_parse()
    test = OfflineEvaluation(options.task_name, options.targetPath,
                             options.armResultPath, options.config_path)
    if options.flag == 0:
        test.pc_test(options.model, options.weights)
    elif options.flag == 1:
        test.arm_test()
    elif options.flag == 2:
        test.pc_arm_test(options.model, options.weights)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()

