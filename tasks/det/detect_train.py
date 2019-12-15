#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import time
from data_loader.det.detection_train_dataloader import DetectionTrainDataloader
from torch_utility.torch_model_process import TorchModelProcess
from solver.torch_optimizer import TorchOptimizer
from solver.lr_scheduler import WarmupMultiStepLR
from utility.logger import AverageMeter, Logger
from config import detectConfig
from tasks.det.detect_test import DetectionTest


class DetectionTrain():

    def __init__(self, cfg_path, gpu_id):
        if not os.path.exists(detectConfig.snapshotPath):
            os.makedirs(detectConfig.snapshotPath, exist_ok=True)

        self.logger = Logger(os.path.join("./log", "logs"))
        self.lossTrain = AverageMeter()

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(detectConfig.optimizerConfig)
        self.multi_lr = WarmupMultiStepLR(detectConfig.base_lr, [[50, 1], [70, 0.1], [100, 0.01]],
                                          warm_epoch=0, warmup_iters=1000)
        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.avg_loss = -1
        self.start_epoch = 0
        self.best_mAP = 0

        self.detect_test = DetectionTest(cfg_path, gpu_id)

    def load_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.best_mAP = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model, detectConfig.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):
        dataloader = DetectionTrainDataloader(train_path, detectConfig.className,
                                              detectConfig.train_batch_size, detectConfig.imgSize,
                                              multi_scale=False, augment=True, balanced_sample=True)
        total_images = len(dataloader)
        self.load_param(detectConfig.latest_weights_file)

        t0 = time.time()
        for epoch in range(self.start_epoch, detectConfig.maxEpochs):
            # self.optimizer = self.torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for i, (images, targets) in enumerate(dataloader):
                current_iter = epoch * total_images + i
                lr = self.multi_lr.get_lr(epoch, current_iter)
                self.multi_lr.adjust_learning_rate(self.optimizer, lr)
                if sum([len(x) for x in targets]) < 1:  # if no targets continue
                    continue

                # Compute loss, compute gradient, update parameters
                output_list = self.model(images.to(self.device))
                loss = self.compute_loss(output_list, targets)
                loss.backward()

                # accumulate gradient for x batches before optimizing
                if ((i + 1) % detectConfig.accumulated_batches == 0) or (i == len(dataloader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.avg_loss < 0:
                    self.avg_loss = (loss.cpu().detach().numpy() / detectConfig.train_batch_size)
                self.avg_loss = 0.9 * (loss.cpu().detach().numpy() / detectConfig.train_batch_size) + 0.1 * self.avg_loss
                print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, i, len(dataloader),
                                                                                    '%.3f' % self.avg_loss,
                                                                                    '%.7f' % self.optimizer.param_groups[0][
                                                                                        'lr'], time.time() - t0))
                self.update_logger(current_iter, loss.data)

                t0 = time.time()

            self.test(val_path, epoch)

    def update_logger(self, step, value):
        self.lossTrain.update(value)
        if (step % detectConfig.display) == 0:
            # print(lossTrain.avg)
            self.logger.scalar_summary("losstrain", self.lossTrain.avg, step)
            self.lossTrain.reset()

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        for k in range(0, loss_count):
            loss += self.model.lossList[k](output_list[k], targets)
        return loss

    def test(self, val_path, epoch):
        save_model_path = os.path.join(detectConfig.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.best_mAP)
        self.detect_test.load_weights(save_model_path)
        mAP, aps = self.detect_test.test(val_path)
        self.detect_test.save_test_result(epoch, mAP, aps)
        self.best_mAP = self.torchModelProcess.saveBestModel(mAP, save_model_path,
                                                             detectConfig.best_weights_file)