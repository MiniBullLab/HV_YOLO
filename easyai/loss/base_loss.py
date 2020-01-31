#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from easyai.base_name.loss_name import LossType


class BaseLoss(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.loss_name = name

    def set_name(self, name):
        self.loss_name = name

    def get_name(self):
        return self.loss_name
