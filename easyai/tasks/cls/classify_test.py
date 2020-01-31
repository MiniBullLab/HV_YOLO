#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.data_loader.cls.classify_dataloader import get_classify_val_dataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.evaluation.classify_accuracy import ClassifyAccuracy
from easyai.config import classify_config


class ClassifyTest():

    def __init__(self, cfg_path, gpu_id):
        self.torchModelProcess = TorchModelProcess()
        self.evaluation = ClassifyAccuracy()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.topK = (1, )

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def test(self, val_path):
        dataloader = get_classify_val_dataloader(val_path, classify_config.imgSize, 1)
        for index, (images, labels) in enumerate(dataloader):
            output_list = self.model(images.to(self.device))
            output = self.compute_output(output_list)
            self.evaluation.compute_topk(output.data,
                                         labels.to(self.device),
                                         self.topK)
        self.print_evaluation()
        return self.evaluation.get_top1()

    def save_test_result(self, epoch):
        # Write epoch results
        save_result_path = os.path.join(classify_config.root_save_dir, 'result.txt')
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

    def compute_output(self, output_list):
        output = None
        if len(output_list) == 1:
            output = self.model.lossList[0](output_list[0])
        return output

    def print_evaluation(self):
        if max(self.topK) > 1:
            print('prec{}: {} \t prec{}: {}\t'.format(self.topK[0],
                                                      self.topK[1],
                                                      self.evaluation.get_top1(),
                                                      self.evaluation.get_topK()))
        else:
            print('prec1: {} \t'.format(self.evaluation.get_top1()))
