#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import time
import torch
import torch.nn as nn
from easyai.data_loader.sr.super_resolution_dataloader import get_sr_dataloader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.torch_utility.torch_vision.torchvision_process import TorchVisionProcess
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.solver.lr_scheduler import MultiStageLR
from easyai.model.sr.MSRResNet import MSRResNet
from easyai.config.task import super_resolution_config
from math import log10

class SuperResolutionTrain():

    def __init__(self):
        if not os.path.exists(super_resolution_config.snapshotPath):
            os.makedirs(super_resolution_config.snapshotPath, exist_ok=True)

        self.torchModelProcess = TorchModelProcess()
        self.torchOptimizer = TorchOptimizer(super_resolution_config.optimizerConfig)
        self.multiLR = MultiStageLR(super_resolution_config.base_lr, [[50, 1], [70, 0.1], [100, 0.01]])
        self.device = self.torchModelProcess.getDevice()
        self.model = MSRResNet(super_resolution_config.in_nc, upscale_factor=super_resolution_config.upscale_factor).to(self.device)

        vision_process = TorchVisionProcess()
        self.input_transform = vision_process.input_transform(super_resolution_config.crop_size,
                                                              super_resolution_config.upscale_factor)
        self.target_transform = vision_process.target_transform(super_resolution_config.crop_size)

        self.start_epoch = 0
        self.psnr = 0

    def load_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            checkpoint = self.torchModelProcess.loadLatestModelWeight(latest_weights_path, self.model)
            self.torchModelProcess.modelTrainInit(self.model)
        else:
            self.torchModelProcess.modelTrainInit(self.model)
        self.start_epoch, self.best_mAP = self.torchModelProcess.getLatestModelValue(checkpoint)

        self.torchOptimizer.createOptimizer(self.start_epoch, self.model, super_resolution_config.base_lr)
        self.optimizer = self.torchOptimizer.getLatestModelOptimizer(checkpoint)

    def update_logger(self, step, value):
        pass

    def compute_loss(self, output, targets):
        loss = 0
        criterion = nn.MSELoss()
        loss += criterion(output, targets)
        return loss

    def test(self, epoch):
        save_model_path = os.path.join(super_resolution_config.snapshotPath, "model_epoch_%d.pt" % epoch)
        self.torchModelProcess.saveLatestModel(save_model_path, self.model,
                                               self.optimizer, epoch, self.psnr)
        testing_data_loader = get_sr_dataloader(super_resolution_config.val_set,
                                                super_resolution_config.val_set,
                                                super_resolution_config.test_batch_size,
                                                num_workers=8,
                                                input_transform=self.input_transform,
                                                target_transform=self.target_transform)
        avg_psnr = 0
        with torch.no_grad():
            for batch in testing_data_loader:
                input, target = batch[0].to(self.device), batch[1].to(self.device)

                prediction = self.model(input)
                mse = self.compute_loss(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

        self.psnr = avg_psnr / len(testing_data_loader)

        self.torchModelProcess.saveBestModel(self.psnr,
                                             save_model_path,
                                             super_resolution_config.best_weights_file)

        print("===> Avg. PSNR: {:.4f} dB".format(self.psnr))

    def train(self):
        training_data_loader = get_sr_dataloader(super_resolution_config.train_set,
                                                 super_resolution_config.train_set,
                                                 super_resolution_config.train_batch_size,
                                                 num_workers=8,
                                                 shuffle=True,
                                                 input_transform=self.input_transform,
                                                 target_transform=self.target_transform)
        total_images = len(training_data_loader)
        self.load_param(super_resolution_config.latest_weights_file)

        t0 = time.time()
        for epoch in range(self.start_epoch, super_resolution_config.maxEpochs):
            self.optimizer.zero_grad()
            for i, batch in enumerate(training_data_loader, 1):
                imgs, targets = batch[0].to(self.device), batch[1].to(self.device)

                current_iter = epoch * total_images + i
                lr = self.multiLR.get_lr(epoch, current_iter)
                self.multiLR.adjust_learning_rate(self.optimizer, lr)

                # Compute loss, compute gradient, update parameters
                output = self.model(imgs.to(self.device))
                loss = self.compute_loss(output, targets)
                loss.backward()

                # accumulate gradient for x batches before optimizing
                if ((i + 1) % super_resolution_config.accumulated_batches == 0) or (i == len(training_data_loader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, i, len(training_data_loader),
                                                                                    '%.3f' % loss,
                                                                                    '%.7f' %
                                                                                    self.optimizer.param_groups[0][
                                                                                        'lr'], time.time() - t0))
                self.update_logger(current_iter, loss.data)

                t0 = time.time()

            self.test(epoch)


def main():
    print("process start...")
    detect_train = SuperResolutionTrain()
    detect_train.train()
    print("process end!")


if __name__ == '__main__':
    main()
