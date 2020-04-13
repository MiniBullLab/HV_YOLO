#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.data_loader.pc_cls.classify_pc_dataloader import get_classify_train_dataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.torch_utility.torch_freeze_bn import TorchFreezeNormalization
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.solver.lr_factory import LrSchedulerFactory
from easyai.utility.train_log import TrainLogger
from easyai.tasks.utility.base_task import DelayedKeyboardInterrupt
from easyai.tasks.utility.base_train import BaseTrain
from easyai.tasks.pc_cls.pc_classify_test import PointCloudClassifyTest
from easyai.base_name.task_name import TaskName


class PointCloudClassifyTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path)
        self.set_task_name(TaskName.PC_Classify_Task)
        self.train_task_config = self.config_factory.get_config(self.task_name, self.config_path)

        self.train_logger = TrainLogger(self.train_task_config.log_name,
                                        self.train_task_config.root_save_dir)
        self.torchModelProcess = TorchModelProcess()
        self.freeze_normalization = TorchFreezeNormalization()
        self.torchOptimizer = TorchOptimizer(self.train_task_config.optimizer_config)

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

        self.classify_test = PointCloudClassifyTest(cfg_path, gpu_id, config_path)

        self.total_clouds = 0
        self.start_epoch = 0
        self.best_precision = 0
        self.optimizer = None

    def load_pretrain_model(self, weights_path):
        self.torchModelProcess.loadPretainModel(weights_path, self.model)

    def load_latest_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path is not None and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.model = self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.model = self.torchModelProcess.modelTrainInit(self.model)

        self.start_epoch, self.best_precision = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.freeze_optimizer_layer(self.start_epoch,
                                                   self.train_task_config.base_lr,
                                                   self.model,
                                                   self.train_task_config.freeze_layer_name,
                                                   self.train_task_config.freeze_layer_type)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def train(self, train_path, val_path):

        dataloader = get_classify_train_dataloader(train_path,
                                                   self.train_task_config.number_point_features,
                                                   self.train_task_config.train_batch_size,
                                                   self.train_task_config.train_data_augment)

        self.total_clouds = len(dataloader)

        lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                        self.train_task_config.max_epochs,
                                        self.total_clouds)
        lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.lr_scheduler_config)

        self.load_latest_param(self.train_task_config.latest_weights_file)

        self.train_task_config.save_config()
        self.timer.tic()
        self.model.train()
        self.freeze_normalization.freeze_normalization_layer(self.model,
                                                             self.train_task_config.freeze_bn_layer_name,
                                                             self.train_task_config.freeze_bn_type)
        try:
            for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
                # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
                self.optimizer.zero_grad()
                for idx, (clouds, targets) in enumerate(dataloader):
                    current_iter = epoch * self.total_clouds + idx
                    lr = lr_scheduler.get_lr(epoch, current_iter)
                    lr_scheduler.adjust_learning_rate(self.optimizer, lr)
                    loss = self.compute_backward(clouds, targets, idx)
                    self.update_logger(idx, self.total_clouds, epoch, loss)

                save_model_path = self.save_train_model(epoch)
                self.test(val_path, epoch, save_model_path)
        except Exception as e:
            raise e
        finally:
            self.train_logger.close()

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss = self.compute_loss(output_list, targets)
        loss.backward()
        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) or \
                (setp_index == self.total_clouds - 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
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
        self.train_logger.train_log(step, loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)

        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.7f' % loss_value,
                                                                            '%.7f' % lr,
                                                                            '%.5f' % self.timer.toc(True)))

    def save_train_model(self, epoch):
        with DelayedKeyboardInterrupt():
            self.train_logger.epoch_train_log(epoch)
            if self.train_task_config.is_save_epoch_model:
                save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                               "pc_cls_model_epoch_%d.pt" % epoch)
            else:
                save_model_path = self.train_task_config.latest_weights_file
            self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                                   self.optimizer, epoch,
                                                   self.best_precision)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        self.classify_test.load_weights(save_model_path)
        precision = self.classify_test.test(val_path)
        self.classify_test.save_test_value(epoch)

        # save best model
        self.best_precision = self.torchModelProcess.saveBestModel(precision,
                                                                   save_model_path,
                                                                   self.train_task_config.best_weights_file)
