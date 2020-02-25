#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.data_loader.det.detection_train_dataloader import DetectionTrainDataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.solver.lr_scheduler import WarmupMultiStepLR
from easyai.utility.train_log import TrainLogger
from easyai.tasks.utility.base_train import BaseTrain
from easyai.tasks.det2d.detect2d_test import Detection2dTest
from easyai.config import detect_config


class Detection2dTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id):
        super().__init__()
        if not os.path.exists(detect_config.snapshotPath):
            os.makedirs(detect_config.snapshotPath, exist_ok=True)

        self.train_logger = TrainLogger(detect_config.log_name)

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(detect_config.optimizerConfig)
        self.multi_lr = WarmupMultiStepLR(detect_config.base_lr, [[50, 1], [70, 0.1], [100, 0.01]],
                                          warm_epoch=0, warmup_iters=1000)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.detect_test = Detection2dTest(cfg_path, gpu_id)

        self.total_images = 0
        self.optimizer = None
        self.avg_loss = -1
        self.start_epoch = 0
        self.best_mAP = 0

    def load_pretrain_model(self, weights_path):
        pass

    def load_latest_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.best_mAP = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model, detect_config.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):
        dataloader = DetectionTrainDataloader(train_path, detect_config.className,
                                              detect_config.train_batch_size, detect_config.imgSize,
                                              multi_scale=False, augment=True, balanced_sample=True)
        self.total_images = len(dataloader)
        self.load_param(detect_config.latest_weights_file)

        self.timer.tic()
        for epoch in range(self.start_epoch, detect_config.maxEpochs):
            # self.optimizer = self.torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for i, (images, targets) in enumerate(dataloader):
                current_iter = epoch * self.total_images + i
                lr = self.multi_lr.get_lr(epoch, current_iter)
                self.multi_lr.adjust_learning_rate(self.optimizer, lr)
                if sum([len(x) for x in targets]) < 1:  # if no targets continue
                    continue
                loss = self.compute_backward(images, targets, i)
                self.update_logger(i, self.total_images, epoch, loss)

            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss = self.compute_loss(output_list, targets)
        loss.backward()

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % detect_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        for k in range(0, loss_count):
            loss += self.model.lossList[k](output_list[k], targets)
        return loss

    def update_logger(self, index, total, epoch, loss):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss.data

        if self.avg_loss < 0:
            self.avg_loss = (loss.cpu().detach().numpy() / detect_config.train_batch_size)
        self.avg_loss = 0.9 * (loss.cpu().detach().numpy() / detect_config.train_batch_size) + 0.1 * self.avg_loss

        self.train_logger.train_log(step, loss_value, detect_config.display)
        self.train_logger.lr_log(step, lr, detect_config.display)
        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.3f' % self.avg_loss,
                                                                            '%.7f' % lr,
                                                                            self.timer.toc(True)))

    def save_train_model(self, epoch):
        self.train_logger.epoch_train_log(epoch)
        save_model_path = os.path.join(detect_config.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.best_mAP)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        self.detect_test.load_weights(save_model_path)
        mAP, aps = self.detect_test.test(val_path)
        self.detect_test.save_test_value(epoch, mAP, aps)

        self.best_mAP = self.torchModelProcess.saveBestModel(mAP, save_model_path,
                                                             detect_config.best_weights_file)
