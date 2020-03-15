#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.cls.classify_dataloader import get_classify_val_dataloader
from easyai.tasks.cls.classify import Classify
from easyai.evaluation.classify_accuracy import ClassifyAccuracy
from easyai.config.classify_config import ClassifyConfig


class ClassifyTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__()
        self.classify_config = ClassifyConfig()
        self.classify_config.load_config(config_path)
        self.classify_inference = Classify(cfg_path, gpu_id)
        self.topK = (1,)
        self.evaluation = ClassifyAccuracy(top_k=self.topK)

    def load_weights(self, weights_path):
        self.classify_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_classify_val_dataloader(val_path,
                                                 self.classify_config.data_mean,
                                                 self.classify_config.data_std,
                                                 self.classify_config.image_size,
                                                 self.classify_config.test_batch_size)
        self.evaluation.clean_data()
        for index, (images, labels) in enumerate(dataloader):
            output = self.classify_inference.infer(images)
            self.evaluation.torch_eval(output.data, labels.to(output.device))
        self.print_evaluation()
        return self.evaluation.get_top1()

    def save_test_value(self, epoch):
        # Write epoch results
        save_result_path = self.classify_config.test_result_path
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
