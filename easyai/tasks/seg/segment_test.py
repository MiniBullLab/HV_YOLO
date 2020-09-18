#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.seg.segment_dataloader import get_segment_val_dataloader
from easyai.tasks.seg.segment import Segmentation
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.evaluation.segmention_metric import SegmentionMetric
from easyai.helper.average_meter import AverageMeter
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Segment_Task)
class SegmentionTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Segment_Task)
        self.segment_inference = Segmentation(cfg_path, gpu_id, config_path)
        self.model = self.segment_inference.model
        self.device = self.segment_inference.device

        self.output_process = SegmentResultProcess()

        self.epoch_loss_average = AverageMeter()

        self.metric = SegmentionMetric(len(self.test_task_config.segment_class))
        self.threshold = 0.5  # binary class threshold

    def load_weights(self, weights_path):
        self.segment_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_segment_val_dataloader(val_path, self.test_task_config)
        print("Eval data num: {}".format(len(dataloader)))
        self.timer.tic()
        self.metric.reset()
        self.epoch_loss_average.reset()
        for i, (images, segment_targets) in enumerate(dataloader):
            prediction, output_list = self.segment_inference.infer(images, self.threshold)
            loss = self.compute_loss(output_list, segment_targets)
            gt = segment_targets[0].data.cpu().numpy()
            self.metric.numpy_eval(prediction, gt)
            self.metirc_loss(i, loss)

        score, class_score = self.metric.get_score()
        average_loss = self.epoch_loss_average.avg
        self.print_evaluation(score)
        return score, class_score, average_loss

    def save_test_value(self, epoch, score, class_score):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
            for i, iou in class_score.items():
                file.write(self.test_task_config.segment_class[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                output, target = self.output_process.output_feature_map_resize(output_list[0], targets)
                loss = self.model.lossList[0](output, target)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.lossList[0](output_list, targets)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    output, target = self.output_process.output_feature_map_resize(output_list[k], targets)
                    loss += self.model.lossList[k](output, target)
            else:
                print("compute loss error")
        return loss

    def metirc_loss(self, step, loss):
        loss_value = loss.data.cpu().squeeze()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {} | Time: {}".format(step,
                                                        loss_value,
                                                        self.timer.toc(True)))

    def print_evaluation(self, score):
        for k, v in score.items():
            print(k, v)
