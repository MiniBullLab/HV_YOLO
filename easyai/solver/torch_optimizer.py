#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import re
import torch
from easyai.base_name.block_name import BlockType


class TorchOptimizer():
    def __init__(self, config):
        self.optimizers = {
            'SGD': torch.optim.SGD,
            'ASGD': torch.optim.ASGD,
            'Adam': torch.optim.Adam,
            'Adamax': torch.optim.Adamax,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'Rprop': torch.optim.Rprop,
            'RMSprop': torch.optim.RMSprop
        }
        self.config = config
        self.optimizer = None

    def freeze_optimizer_layer(self, epoch, base_lr, model,
                               layer_name, flag=0):
        if flag == 0:
            pass
        elif flag == 1:
            layer_name = layer_name.strip()
            self.freeze_front_layer(model, layer_name)
        elif flag == 2:
            layer_name = layer_name.strip()
            for key, block in model._modules.items():
                if key == BlockType.BaseNet:
                    self.freeze_front_layer(block, layer_name)
                    break
        elif flag == 3:
            layer_names = [x.strip() for x in layer_name.split(',') if x.strip()]
            self.freeze_layers(model, layer_names)
        elif flag == 4:
            layer_names = [x.strip() for x in layer_name.split(',') if x.strip()]
            for key, block in model._modules.items():
                if key == BlockType.BaseNet:
                    self.freeze_layers(block, layer_names)
                    break
        elif flag == 5:
            layer_name = layer_name.strip()
            self.freeze_layer_from_name(model, layer_name)
        elif flag == 6:
            layer_name = layer_name.strip()
            for key, block in model._modules.items():
                if key == BlockType.BaseNet:
                    self.freeze_layer_from_name(block, layer_name)
                    break
        else:
            print("freeze layer error")
        self.createOptimizer(epoch, model, base_lr)

    def createOptimizer(self, epoch, model, base_lr):
        em = 0
        for e in self.config.keys():
            if epoch >= e:
                em = e
        setting = self.config[em]
        self.optimizer = self.optimizers[setting['optimizer']](
            filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
        self.adjust_param(self.optimizer, setting)

    def freeze_layers(self, model, layer_names):
        for key, block in model._modules.items():
            if key in layer_names:
                for param in block.parameters():
                    param.requires_grad = False

    def freeze_layer_from_name(self, model, layer_name):
        layer_name_re = None
        if layer_name is not None:
            layer_name_re = re.compile(layer_name)
        for key, block in model._modules.items():
            if layer_name_re.match(key) is not None:
                for param in block.parameters():
                    param.requires_grad = False

    def freeze_front_layer(self, model, layer_name):
        for key, block in model._modules.items():
            for param in block.parameters():
                param.requires_grad = False
            if layer_name == key:
                break

    def print_freeze_layer(self, model):
        for key, block in model._modules.items():
            print(key)
            for param in block.named_parameters():
                print(param[0], param[1].requires_grad)

    def adjust_optimizer(self, epoch, lr):
        # select the true epoch to adjust the optimizer
        em = 0
        for e in self.config.keys():
            if epoch >= e:
                em = e
        setting = self.config[em]
        self.optimizer = self.modify_optimizer(self.optimizer, setting)
        self.adjust_param(self.optimizer, setting)
        return self.optimizer

    def getLatestModelOptimizer(self, checkpoint):
        if checkpoint is not None:
            if checkpoint.get('optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.optimizer

    def modify_optimizer(self, optimizer, setting):
        result = None
        if 'optimizer' in setting:
            result = self.optimizers[setting['optimizer']](optimizer.param_groups)
            print('OPTIMIZER - setting method = %s' % setting['optimizer'])
        return result

    def adjust_param(self, optimizer, setting):
        for i_group, param_group in enumerate(optimizer.param_groups):
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
                    print('OPTIMIZER - group %s setting %s = %s' %
                          (i_group, key, param_group[key]))


if __name__ == "__main__":
    pass
