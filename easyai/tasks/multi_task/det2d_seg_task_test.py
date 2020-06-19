#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.calculate_mAp import CalculateMeanAp
from easyai.data_loader.multi_task.det2d_seg_val_dataloader import get_det2d_seg_val_dataloader
from easyai.tasks.multi_task.det2d_seg_task import Det2dSegTask
from easyai.evaluation.segmention_metric import SegmentionMetric
from easyai.base_name.task_name import TaskName


class Det2dSegTaskTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Det2d_Seg_Task)
        self.multi_task_inference = Det2dSegTask(cfg_path, gpu_id, config_path)
        self.threshold_det = 5e-3
        self.threshold_seg = 0.5

    def load_weights(self, weights_path):
        self.multi_task_inference.load_weights(weights_path)

    def test(self, val_path):
        os.system('rm -rf ' + self.test_task_config.save_result_dir)
        os.makedirs(self.test_task_config.save_result_dir, exist_ok=True)

        dataloader = get_det2d_seg_val_dataloader(val_path,
                                                  self.test_task_config.detect_name,
                                                  self.test_task_config.segment_name,
                                                  image_size=self.test_task_config.image_size,
                                                  data_channel=self.test_task_config.image_channel,
                                                  batch_size=1)

        det2d_evaluator = CalculateMeanAp(val_path, self.test_task_config.detect_name)
        seg_metric = SegmentionMetric(len(self.test_task_config.segment_name))

        self.timer.tic()
        seg_metric.reset()
        for i, (image_path, src_image, input_image, segment_targets) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            self.multi_task_inference.set_src_size(src_image.numpy()[0])

            result_dets, result_seg = self.multi_task_inference.infer(input_image,
                                                                      threshold_det=self.threshold_det,
                                                                      threshold_seg=self.threshold_seg)
            detection_objects, _ = self.multi_task_inference.postprocess((result_dets, result_seg))

            gt = segment_targets[0].data.cpu().numpy()
            seg_metric.eval(result_seg, gt)

            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))

            path, filename_post = os.path.split(image_path[0])
            self.multi_task_inference.save_result(filename_post, detection_objects)

        mAP, aps = det2d_evaluator.eval(self.test_task_config.save_result_dir)
        score, class_score = seg_metric.get_score()
        self.print_evaluation(score)
        return mAP, aps, score, class_score

    def save_test_value(self, epoch, mAP, aps, score, class_score):
        # Write epoch results
        with open(self.test_task_config.save_evaluation_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | mIoU: {:.3f} | ".format(epoch, mAP, score['Mean IoU : \t']))
            for i, ap in enumerate(aps):
                file.write(self.test_task_config.detect_name[i] + ": {:.3f} ".format(ap))
            for i, iou in class_score.items():
                file.write(self.test_task_config.segment_name[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

    def print_evaluation(self, score):
        for k, v in score.items():
            print(k, v)
