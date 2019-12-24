#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import time
from data_loader.seg.segment_dataloader import get_segment_train_dataloader
from solver.lr_scheduler import PolyLR
from solver.torch_optimizer import TorchOptimizer
from torch_utility.torch_model_process import TorchModelProcess
from utility.train_log import TrainLogger
from tasks.seg.segment_test import SegmentionTest
from config import segment_config


class SegmentionTrain():

    def __init__(self, cfg_path, gpu_id):
        if not os.path.exists(segment_config.snapshotPath):
            os.makedirs(segment_config.snapshotPath, exist_ok=True)

        self.train_logger = TrainLogger(segment_config.log_name)

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(segment_config.optimizerConfig)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.segment_test = SegmentionTest(cfg_path, gpu_id)

        self.optimizer = None
        self.start_epoch = 0
        self.bestmIoU = 0

    def load_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.bestmIoU = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model, segment_config.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):
        dataloader = get_segment_train_dataloader(train_path, segment_config.imgSize,
                                                  segment_config.train_batch_size)
        total_images = len(dataloader)
        self.load_param(segment_config.latest_weights_file)

        # set learning policy
        total_iteration = segment_config.maxEpochs * len(dataloader)
        polyLR = PolyLR(segment_config.base_lr, total_iteration, segment_config.lr_power)

        t0 = time.time()
        for epoch in range(self.start_epoch, segment_config.maxEpochs):
            # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for idx, (images, segments) in enumerate(dataloader):

                current_idx = epoch * total_images + idx
                lr = polyLR.get_lr(epoch, current_idx)
                polyLR.adjust_learning_rate(self.optimizer, lr)

                # Compute loss, compute gradient, update parameters
                output_list = self.model(images.to(self.device))
                loss = self.compute_loss(output_list, segments)
                loss.backward()

                # accumulate gradient for x batches before optimizing
                if ((idx + 1) % segment_config.accumulated_batches == 0) \
                        or (idx == total_images - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.update_logger(idx, total_images, epoch, loss.data, t0)
                t0 = time.time()

            self.test(val_path, epoch)

    def update_logger(self, index, total, epoch, loss_value, time_value):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        self.train_logger.train_log(step, loss_value, segment_config.display)
        self.train_logger.lr_log(step, lr, segment_config.display)

        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.3f' % loss_value,
                                                                            '%.7f' % lr,
                                                                            time.time() - time_value))

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        targets = targets.to(self.device)
        for k in range(0, loss_count):
            output, target = self.model.lossList[k].segment_resize(output_list[k], targets)
            loss += self.model.lossList[k](output, target)
        return loss

    def test(self, val_path, epoch):
        self.train_logger.epoch_train_log(epoch)
        save_model_path = os.path.join(segment_config.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.bestmIoU)
        self.segment_test.load_weights(save_model_path)
        score, class_iou = self.segment_test.test(val_path)
        self.segment_test.save_test_result(epoch, score, class_iou)
        self.bestmIoU = self.torchModelProcess.saveBestModel(score['Mean IoU : \t'],
                                                             save_model_path,
                                                             segment_config.best_weights_file)

