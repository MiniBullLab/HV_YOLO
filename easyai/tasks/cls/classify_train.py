#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.data_loader.cls.classify_dataloader import get_classify_train_dataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.solver.lr_scheduler import MultiStageLR, WarmupMultiStepLR
from easyai.utility.train_log import TrainLogger
from easyai.config.classify_config import ClassifyConfig
from easyai.tasks.utility.base_train import BaseTrain
from easyai.tasks.cls.classify_test import ClassifyTest


class ClassifyTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__()

        self.classify_config = ClassifyConfig()
        self.classify_config.load_config(config_path)

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(self.classify_config.optimizer_config)
        # self.multiLR = WarmupMultiStepLR(self.classify_config.base_lr,
        #                                  [[60, 1], [120, 0.2], [160, 0.04], [200, 0.0008]],
        #                                  0, 390)
        self.multiLR = MultiStageLR(self.classify_config.base_lr, [[60, 0.1], [120, 0.02],
                                                                   [160, 0.004], [200, 0.0008]])

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.classify_test = ClassifyTest(cfg_path, gpu_id)

        self.train_logger = TrainLogger(self.classify_config.log_name)

        self.total_images = 0
        self.start_epoch = 0
        self.best_precision = 0
        self.optimizer = None

    def load_pretrain_model(self, weights_path):
        pass

    def load_latest_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path is not None and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.model = self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.model = self.torchModelProcess.modelTrainInit(self.model)

        self.start_epoch, self.best_precision = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model,
                                            self.classify_config.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):

        dataloader = get_classify_train_dataloader(train_path,
                                                   self.classify_config.data_mean,
                                                   self.classify_config.data_std,
                                                   self.classify_config.image_size,
                                                   self.classify_config.train_batch_size)

        self.total_images = len(dataloader)

        self.load_latest_param(self.classify_config.latest_weights_file)
        self.classify_config.save_config()
        self.timer.tic()
        self.model.train()
        for epoch in range(self.start_epoch, self.classify_config.max_epochs):
            # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for idx, (imgs, targets) in enumerate(dataloader):
                current_iter = epoch * self.total_images + idx
                lr = self.multiLR.get_lr(epoch, current_iter)
                self.multiLR.adjust_learning_rate(self.optimizer, lr)
                loss = self.compute_backward(imgs, targets, idx)
                self.update_logger(idx, self.total_images, epoch, loss)

            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

        self.train_logger.close()

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss = self.compute_loss(output_list, targets)
        loss.backward()
        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.classify_config.accumulated_batches == 0) or \
                (setp_index == self.total_images - 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        targets = targets.to(self.device)
        for k in range(0, loss_count):
            loss += self.model.lossList[k](output_list[k], targets)
        return loss

    def update_logger(self, index, total, epoch, loss):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss.data.cpu().squeeze()
        self.train_logger.train_log(step, loss_value, self.classify_config.display)
        self.train_logger.lr_log(step, lr, self.classify_config.display)

        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.3f' % loss_value,
                                                                            '%.7f' %
                                                                            lr,
                                                                            self.timer.toc(True)))

    def save_train_model(self, epoch):
        self.train_logger.epoch_train_log(epoch)
        if self.classify_config.is_save_epoch_model:
            save_model_path = os.path.join(self.classify_config.snapshot_path,
                                           "model_epoch_%d.pt" % epoch)
        else:
            save_model_path = self.classify_config.latest_weights_file
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch,
                                               self.best_precision)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        self.classify_test.load_weights(save_model_path)
        precision = self.classify_test.test(val_path)
        self.classify_test.save_test_value(epoch)

        self.best_precision = self.torchModelProcess.saveBestModel(precision,
                                                                   save_model_path,
                                                                   self.classify_config.best_weights_file)
