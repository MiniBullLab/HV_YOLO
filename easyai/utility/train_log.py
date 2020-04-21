#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import time
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from easyai.helper.average_meter import AverageMeter


class TrainLogger():

    def __init__(self, log_name, log_save_dir):
        current_time = time.strftime("%Y-%m-%dT%H_%M", time.localtime())
        log_name = "%s_%s" % (log_name, current_time)
        self.root_dir = log_save_dir
        self.log_dir = os.path.join(self.root_dir, log_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        self.loss_average = AverageMeter()
        self.lr_average = AverageMeter()
        self.epoch_loss_average = AverageMeter()

    def train_log(self, step, value, display):
        self.loss_average.update(value)
        self.epoch_loss_average.update(value)
        if step % display == 0:
            # print(self.loss_average.avg)
            self.add_scalar("loss", self.loss_average.avg, step)
            self.loss_average.reset()

    def lr_log(self, step, lr, display):
        self.lr_average.update(lr)
        if step % display == 0:
            # print(step, self.lr_average.avg)
            self.add_scalar("lr", self.lr_average.avg, step)
            self.lr_average.reset()

    def epoch_train_log(self, epoch):
        self.add_scalar("train epoch loss", self.epoch_loss_average.avg, epoch)
        self.epoch_loss_average.reset()

    def eval_log(self, tag, epoch, value):
        self.add_scalar(tag, value, epoch)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_image(self, tag, img, step):
        x = vutils.make_grid(img, normalize=True, scale_each=True)
        self.writer.add_image(tag, x, step)

    def write_model(self, model, input_x):
        self.writer.add_graph(model, (input_x,))

    def close(self):
        save_path = os.path.join(self.root_dir, "all_scalars.json")
        self.writer.export_scalars_to_json(save_path)
        self.writer.close()
