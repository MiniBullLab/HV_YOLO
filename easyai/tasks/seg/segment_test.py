#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.seg.segment_dataloader import get_segment_val_dataloader
from easyai.tasks.seg.segment import Segmentation
from easyai.evaluation.segmention_metric import SegmentionMetric
from easyai.config.segment_config import SegmentionConfig


class SegmentionTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__()
        self.segmention_config = SegmentionConfig()
        self.segmention_config.load_config(config_path)
        self.segment_inference = Segmentation(cfg_path, gpu_id)
        self.metric = SegmentionMetric(len(self.segmention_config.class_name))
        self.threshold = 0.5

    def load_weights(self, weights_path):
        self.segment_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_segment_val_dataloader(val_path, self.segmention_config.image_size,
                                                self.segmention_config.test_batch_size)
        print("Eval data num: {}".format(len(dataloader)))
        self.timer.tic()
        self.metric.reset()
        for i, (images, segment_targets) in enumerate(dataloader):
            prediction = self.segment_inference.infer(images, self.threshold)
            gt = segment_targets[0].data.cpu().numpy()
            self.metric.eval(prediction, gt)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))

        score, class_iou = self.metric.get_score()
        self.print_evaluation(score)
        return score, class_iou

    def save_test_value(self, epoch, score, class_iou):
        # write epoch results
        with open(self.segmention_config.save_evaluation_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
            for i, iou in enumerate(class_iou):
                file.write(self.segmention_config.class_name[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

    def print_evaluation(self, score):
        for k, v in score.items():
            print(k, v)
