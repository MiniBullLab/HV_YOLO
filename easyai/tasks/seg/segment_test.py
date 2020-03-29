#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.seg.segment_dataloader import get_segment_val_dataloader
from easyai.tasks.seg.segment import Segmentation
from easyai.evaluation.segmention_metric import SegmentionMetric
from easyai.base_name.task_name import TaskName


class SegmentionTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.Segment_Task)
        self.test_task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.segment_inference = Segmentation(cfg_path, gpu_id)
        self.metric = SegmentionMetric(len(self.test_task_config.class_name))
        self.threshold = 0.5

    def load_weights(self, weights_path):
        self.segment_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_segment_val_dataloader(val_path, self.test_task_config.image_size,
                                                self.test_task_config.test_batch_size)
        print("Eval data num: {}".format(len(dataloader)))
        self.timer.tic()
        self.metric.reset()
        for i, (images, segment_targets) in enumerate(dataloader):
            prediction, output_list = self.segment_inference.infer(images, self.threshold)
            gt = segment_targets[0].data.cpu().numpy()
            self.metric.eval(prediction, gt)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))

        score, class_score = self.metric.get_score()
        self.print_evaluation(score)
        return score, class_score

    def save_test_value(self, epoch, score, class_score):
        # write epoch results
        with open(self.test_task_config.save_evaluation_path, 'a') as file:
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
            for i, iou in class_score.items():
                file.write(self.test_task_config.class_name[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

    def print_evaluation(self, score):
        for k, v in score.items():
            print(k, v)
