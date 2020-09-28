#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.registry import REGISTERED_TRAIN_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class TrainTask():

    def __init__(self, task_name, train_path, val_path,
                 is_convert=False):
        self.task_name = task_name
        self.train_path = train_path
        self.val_path = val_path
        self.is_convert = is_convert
        self.save_onnx_path = None

    def train(self, cfg_path, gpu_id, config_path, pretrain_model_path):
        task_args = {'type': self.task_name,
                     'cfg_path': cfg_path,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        if self.task_name is not None and \
                REGISTERED_TRAIN_TASK.has_class(self.task_name):
            task = build_from_cfg(task_args, REGISTERED_TRAIN_TASK)
            task.load_pretrain_model(pretrain_model_path)
            task.train(self.train_path, self.val_path)
            self.image_model_convert(task, task.model_args)
        else:
            print("This task(%s) not exits!" % self.task_name)

    def image_model_convert(self, train_task, model_args):
        if self.is_convert:
            from easyai.tools.model_tool.model_to_onnx import ModelConverter
            converter = ModelConverter(train_task.train_task_config.image_size)
            self.save_onnx_path = converter.model_convert(model_args,
                                                          train_task.train_task_config.best_weights_path,
                                                          train_task.train_task_config.snapshot_dir)


def main():
    print("process start...")
    options = TaskArgumentsParse.train_input_parse()
    train_task = TrainTask(options.task_name, options.trainPath, options.valPath)
    train_task.train(options.model, 0, options.config_path, options.pretrainModel)
    print("process end!")


if __name__ == '__main__':
    main()
