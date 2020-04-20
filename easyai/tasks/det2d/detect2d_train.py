#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.data_loader.det.detection_train_dataloader import DetectionTrainDataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.torch_utility.torch_freeze_bn import TorchFreezeNormalization
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.solver.lr_factory import LrSchedulerFactory
from easyai.utility.train_log import TrainLogger
from easyai.tasks.utility.base_train import BaseTrain
from easyai.tasks.det2d.detect2d_test import Detection2dTest
from easyai.base_name.task_name import TaskName


class Detection2dTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.Detect2d_Task)
        self.train_task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.train_logger = TrainLogger(self.train_task_config.log_name,
                                        self.train_task_config.root_save_dir)

        self.torchModelProcess = TorchModelProcess()
        self.freeze_normalization = TorchFreezeNormalization()
        self.torchOptimizer = TorchOptimizer(self.train_task_config.optimizer_config)

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.detect_test = Detection2dTest(cfg_path, gpu_id, config_path)

        self.total_images = 0
        self.optimizer = None
        self.avg_loss = -1
        self.start_epoch = 0
        self.best_mAP = 0

    def load_pretrain_model(self, weights_path):
        self.torchModelProcess.loadPretainModel(weights_path, self.model)

    def load_latest_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.model = self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.model = self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.best_mAP = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.freeze_optimizer_layer(self.start_epoch,
                                                   self.train_task_config.base_lr,
                                                   self.model,
                                                   self.train_task_config.freeze_layer_name,
                                                   self.train_task_config.freeze_layer_type)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):
        dataloader = DetectionTrainDataloader(train_path, self.train_task_config.class_name,
                                              self.train_task_config.train_batch_size,
                                              self.train_task_config.image_size,
                                              multi_scale=self.train_task_config.train_multi_scale,
                                              is_augment=self.train_task_config.train_data_augment,
                                              balanced_sample=self.train_task_config.balanced_sample)
        self.total_images = len(dataloader)

        lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                        self.train_task_config.max_epochs,
                                        self.total_images)
        lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.lr_scheduler_config)

        self.load_latest_param(self.train_task_config.latest_weights_file)

        self.train_task_config.save_config()
        self.timer.tic()
        self.model.train()
        self.freeze_normalization.freeze_normalization_layer(self.model,
                                                             self.train_task_config.freeze_bn_layer_name,
                                                             self.train_task_config.freeze_bn_type)
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            # self.optimizer = self.torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for i, (images, targets) in enumerate(dataloader):
                current_iter = epoch * self.total_images + i
                lr = lr_scheduler.get_lr(epoch, current_iter)
                lr_scheduler.adjust_learning_rate(self.optimizer, lr)
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
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            loss = self.model.lossList[0](output_list[0], targets)
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, targets)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                loss += self.model.lossList[k](output_list[k], targets)
        else:
            print("compute loss error")
        return loss

    def update_logger(self, index, total, epoch, loss):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss.data.cpu().squeeze()

        if self.avg_loss < 0:
            self.avg_loss = (loss.cpu().detach().numpy() / self.train_task_config.train_batch_size)
        self.avg_loss = 0.9 * (loss.cpu().detach().numpy() / self.train_task_config.train_batch_size) \
                        + 0.1 * self.avg_loss

        self.train_logger.train_log(step, loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)
        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.7f' % self.avg_loss,
                                                                            '%.7f' % lr,
                                                                            '%.5f' % self.timer.toc(True)))

    def save_train_model(self, epoch):
        self.train_logger.epoch_train_log(epoch)
        if self.train_task_config.is_save_epoch_model:
            save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                           "det2d_model_epoch_%d.pt" % epoch)
        else:
            save_model_path = self.train_task_config.latest_weights_file
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.best_mAP)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            self.detect_test.load_weights(save_model_path)
            mAP, aps = self.detect_test.test(val_path)
            self.detect_test.save_test_value(epoch, mAP, aps)
            # save best model
            self.best_mAP = self.torchModelProcess.saveBestModel(mAP, save_model_path,
                                                                 self.train_task_config.best_weights_file)
        else:
            print("no test!")