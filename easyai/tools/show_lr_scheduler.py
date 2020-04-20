#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.solver.lr_factory import LrSchedulerFactory
from easyai.config.utility.config_factory import ConfigFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class ShowLrScheduler():

    def __init__(self, task_name, config_path=None, epoch_iteration=4000):
        self.epoch_iteration = epoch_iteration
        self.config_factory = ConfigFactory()
        self.lr_config = self.config_factory.get_config(task_name, config_path)
        self.lr_factory = LrSchedulerFactory(self.lr_config.base_lr,
                                             self.lr_config.max_epochs,
                                             self.epoch_iteration)
        self.lr_scheduler = self.lr_factory.get_lr_scheduler(self.lr_config.lr_scheduler_config)
        self.torchOptimizer = TorchOptimizer(self.lr_config.optimizer_config)
        self.optimizer = None

    def show(self):
        lr_list = self.get_lr_lists()
        lr_class_name = self.lr_config.lr_scheduler_config['lr_type'].strip()
        self.show_graph(lr_list, self.lr_config.max_epochs, self.epoch_iteration, lr_class_name)

    def get_lr_lists(self):
        lr_list = []
        model = nn.Linear(1280, 1000)
        self.torchOptimizer.createOptimizer(0, model, self.lr_config.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(None)
        for epoch in range(0, self.lr_config.max_epochs):
            for idx in range(0, self.epoch_iteration):
                current_iter = epoch * self.epoch_iteration + idx
                lr = self.lr_scheduler.get_lr(epoch, current_iter)
                self.lr_scheduler.adjust_learning_rate(self.optimizer, lr)
                lr_list.append(self.optimizer.param_groups[0]['lr'])
        return lr_list

    def show_graph(self, lr_lists, epochs, steps, lr_name):
        plt.clf()
        plt.rcParams['figure.figsize'] = [20, 5]
        x = list(range(epochs * steps))
        plt.plot(x, lr_lists, label=lr_name)
        plt.plot()

        plt.ylim(10e-7, 1)
        plt.yscale("log")
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.title("plot learning rate secheduler {}".format(lr_name))
        plt.legend()
        plt.show()


def main():
    print("process start...")
    options = ToolArgumentsParse.show_lr_parse()
    show_lr = ShowLrScheduler(options.task_name, options.config_path, options.epoch_iteration)
    show_lr.show()
    print("process end!")


if __name__ == '__main__':
    main()