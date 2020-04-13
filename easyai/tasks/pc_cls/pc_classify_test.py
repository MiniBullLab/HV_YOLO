#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.pc_cls.classify_pc_dataloader import get_classify_val_dataloader
from easyai.tasks.pc_cls.pc_classify import PointCloudClassify
from easyai.evaluation.classify_accuracy import ClassifyAccuracy
from easyai.base_name.task_name import TaskName


class PointCloudClassifyTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.PC_Classify_Task)
        self.test_task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.classify_inference = PointCloudClassify(cfg_path, gpu_id, config_path)
        self.topK = (1,)
        self.evaluation = ClassifyAccuracy(top_k=self.topK)

    def load_weights(self, weights_path):
        self.classify_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_classify_val_dataloader(val_path,
                                                 self.test_task_config.number_point_features,
                                                 self.test_task_config.test_batch_size)
        self.evaluation.clean_data()
        for index, (clouds, labels) in enumerate(dataloader):
            output = self.classify_inference.infer(clouds)
            self.evaluation.torch_eval(output.data, labels.to(output.device))
        self.print_evaluation()
        return self.evaluation.get_top1()

    def save_test_value(self, epoch):
        # Write epoch results
        save_result_path = self.test_task_config.test_result_path
        if max(self.topK) > 1:
            with open(save_result_path, 'a') as file:
                file.write("Epoch: {} | prec{}: {:.3f} | prec{}: {:.3f}\n".format(epoch,
                                                                                  self.topK[0],
                                                                                  self.topK[1],
                                                                                  self.evaluation.get_top1(),
                                                                                  self.evaluation.get_topK()))
        else:
            with open(save_result_path, 'a') as file:
                file.write("Epoch: {} | prec1: {:.3f}\n".format(epoch, self.evaluation.get_top1()))

    def print_evaluation(self):
        if max(self.topK) > 1:
            print('prec{}: {} \t prec{}: {}\t'.format(self.topK[0],
                                                      self.topK[1],
                                                      self.evaluation.get_top1(),
                                                      self.evaluation.get_topK()))
        else:
            print('prec1: {} \t'.format(self.evaluation.get_top1()))

