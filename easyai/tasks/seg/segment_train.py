#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
from easyai.data_loader.seg.segment_dataloader import get_segment_train_dataloader
from easyai.solver.lr_scheduler import PolyLR
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.utility.train_log import TrainLogger
from easyai.tasks.utility.base_train import BaseTrain
from easyai.tasks.seg.segment_test import SegmentionTest
from easyai.config import segment_config


class SegmentionTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id):
        super().__init__()
        if not os.path.exists(segment_config.snapshotPath):
            os.makedirs(segment_config.snapshotPath, exist_ok=True)

        self.train_logger = TrainLogger(segment_config.log_name)

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(segment_config.optimizerConfig)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.segment_test = SegmentionTest(cfg_path, gpu_id)

        self.total_images = 0
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

        if segment_config.enable_freeze_layer:
            self.torchOptimizer.freeze_optimizer_layer(self.start_epoch, self.model,
                                                       segment_config.base_lr,
                                                       segment_config.freeze_layer_name)
        else:
            self.torchOptimizer.createOptimizer(self.start_epoch, self.model, segment_config.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):
        dataloader = get_segment_train_dataloader(train_path, segment_config.imgSize,
                                                  segment_config.train_batch_size)
        self.total_images = len(dataloader)
        self.load_param(segment_config.latest_weights_file)

        # set learning policy
        total_iteration = segment_config.maxEpochs * len(dataloader)
        polyLR = PolyLR(segment_config.base_lr, total_iteration, segment_config.lr_power)

        self.timer.tic()
        for epoch in range(self.start_epoch, segment_config.maxEpochs):
            # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for idx, (images, segments) in enumerate(dataloader):
                current_idx = epoch * self.total_images + idx
                lr = polyLR.get_lr(epoch, current_idx)
                polyLR.adjust_learning_rate(self.optimizer, lr)
                loss = self.compute_backward(images, segments, idx)
                self.update_logger(idx, self.total_images, epoch, loss.data)

            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss = self.compute_loss(output_list, targets)
        loss.backward()

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % segment_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        targets = targets.to(self.device)
        for k in range(0, loss_count):
            output, target = self.segment_data_resize(output_list[k], targets)
            loss += self.model.lossList[k](output, target)
        return loss

    def segment_data_resize(self, input_data, target):
        target = target.type(input_data.dtype)
        n, c, h, w = input_data.size()
        nt, ht, wt = target.size()
        # Handle inconsistent size between input and target
        if h > ht and w > wt:  # upsample labels
            target = target.unsequeeze(1)
            target = torch.nn.functional.upsample(target, size=(h, w), mode='nearest')
            target = target.sequeeze(1)
        elif h < ht and w < wt:  # upsample images
            input_data = torch.nn.functional.upsample(input_data, size=(ht, wt), mode='bilinear')
        elif h == ht and w == wt:
            pass
        else:
            print("input_data: (%d,%d) and target: (%d,%d) error "
                  % (h, w, ht, wt))
            raise Exception("segment_data_resize error")
        return input_data, target

    def update_logger(self, index, total, epoch, loss_value):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        self.train_logger.train_log(step, loss_value, segment_config.display)
        self.train_logger.lr_log(step, lr, segment_config.display)

        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.3f' % loss_value,
                                                                            '%.7f' % lr,
                                                                            self.timer.toc(True)))

    def save_train_model(self, epoch):
        self.train_logger.epoch_train_log(epoch)
        save_model_path = os.path.join(segment_config.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.bestmIoU)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        self.segment_test.load_weights(save_model_path)
        score, class_iou = self.segment_test.test(val_path)
        self.segment_test.save_test_value(epoch, score, class_iou)

        self.bestmIoU = self.torchModelProcess.saveBestModel(score['Mean IoU : \t'],
                                                             save_model_path,
                                                             segment_config.best_weights_file)

