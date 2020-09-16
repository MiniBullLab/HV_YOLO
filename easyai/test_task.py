#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.registry import REGISTERED_TEST_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class TestTask():

    def __init__(self, task_name, val_path):
        self.val_path = val_path
        self.task_name = task_name

    def test(self, cfg_path, gpu_id, weight_path, config_path):
        task_args = {'type': self.task_name,
                     'cfg_path': cfg_path,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        if REGISTERED_TEST_TASK.has_class(self.task_name):
            task = build_from_cfg(task_args, REGISTERED_TEST_TASK)
            task.load_weights(weight_path)
            task.test(self.val_path)
        else:
            print("This task(%s) not exits!" % self.task_name)


def main():
    print("process start...")
    options = TaskArgumentsParse.test_input_parse()
    test_task = TestTask(options.task_name, options.valPath)
    test_task.test(options.model, 0, options.weights, options.config_path)
    print("process end!")


if __name__ == '__main__':
    main()
