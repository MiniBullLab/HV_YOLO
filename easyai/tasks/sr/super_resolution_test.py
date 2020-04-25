#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from math import log10
import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.sr.super_resolution_dataloader import get_sr_val_dataloader
from easyai.tasks.sr.super_resolution import SuperResolution
from easyai.helper.average_meter import AverageMeter
from easyai.base_name.task_name import TaskName


class SuperResolutionTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.SuperResolution_Task)
        self.test_task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.sr_inference = SuperResolution(cfg_path, gpu_id, config_path)
        self.model = self.sr_inference.model
        self.device = self.sr_inference.device

        self.epoch_loss_average = AverageMeter()
        self.epoch_avg_psnr = AverageMeter()

    def load_weights(self, weights_path):
        self.sr_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_sr_val_dataloader(val_path, self.test_task_config.image_size,
                                           self.test_task_config.image_channel,
                                           self.test_task_config.upscale_factor,
                                           self.test_task_config.test_batch_size)
        print("Eval data num: {}".format(len(dataloader)))
        self.timer.tic()
        self.epoch_avg_psnr.reset()
        self.epoch_loss_average.reset()
        for i, (images, sr_targets) in enumerate(dataloader):
            prediction, output_list = self.sr_inference.infer(images)
            loss = self.compute_loss(output_list, sr_targets)
            self.compute_metric(loss)
            self.metirc_loss(i, loss)

        score = self.epoch_avg_psnr.avg
        average_loss = self.epoch_loss_average.avg
        self.print_evaluation(score)
        return score, average_loss

    def save_test_value(self, epoch, score):
        # write epoch results
        with open(self.test_task_config.save_evaluation_path, 'a') as file:
            file.write("Epoch: {} | psnr: {:.5f} | ".format(epoch, score))
            file.write("\n")

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                loss = self.model.lossList[0](output_list[0], targets)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    loss += self.model.lossList[k](output_list[k], targets)
            else:
                print("compute loss error")
        return loss

    def metirc_loss(self, step, loss):
        loss_value = loss.data.cpu().squeeze()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {} | Time: {.7f}".format(step,
                                                           loss_value,
                                                           self.timer.toc(True)))

    def compute_metric(self, loss):
        psnr = 10 * log10(1 / loss.item())
        self.epoch_avg_psnr.update(psnr, 1)

    def print_evaluation(self, score):
        print("Average psnr: {.5f}".format(score))
