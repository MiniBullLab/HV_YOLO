#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.key_point_accuracy import KeyPointAccuracy
from easyai.data_loader.key_point2d.key_point2d_dataloader import get_key_points2d_val_dataloader
from easyai.tasks.key_points2d.key_points2d import KeyPoints2d
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.KeyPoints2d_Task)
class KeyPoints2dTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.KeyPoints2d_Task)
        self.inference = KeyPoints2d(cfg_path, gpu_id, config_path)
        self.evaluator = KeyPointAccuracy(self.test_task_config.points_count,
                                          self.test_task_config.points_class)

        self.conf_threshold = 5e-3

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_key_points2d_val_dataloader(val_path,
                                                     self.test_task_config)

        self.timer.tic()
        self.evaluator.reset()
        for i, (images, labels) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            result = self.inference.infer(images, self.conf_threshold)
            labels = labels[0].data.cpu().numpy()
            self.evaluator.eval(result, labels)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))
        accuracy, _ = self.evaluator.get_accuracy()
        return accuracy

    def save_test_value(self, epoch, accuracy):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | accuracy: {:.3f}".format(epoch, accuracy))
            file.write("\n")
