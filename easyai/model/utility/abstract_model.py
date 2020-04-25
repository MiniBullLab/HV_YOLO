#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import torch.nn as nn
import torch.nn.functional as F
import abc


class AbstractModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_name = "None"
        self.block_out_channels = []
        self.index = 0

    @abc.abstractmethod
    def create_block_list(self):
        pass

    def set_name(self, name):
        self.model_name = name

    def get_name(self):
        return self.model_name

    def clear_list(self):
        self.block_out_channels = []
        self.index = 0

    def add_block_list(self, block_name, block, output_channel):
        block_name = "%s_%d" % (block_name, self.index)
        self.add_module(block_name, block)
        self.index += 1
        self.block_out_channels.append(output_channel)

    def print_block_name(self):
        for key in self._modules.keys():
            print(key)
