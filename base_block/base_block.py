#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BaseBlock(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.block_name = name

    def set_name(self, name):
        self.block_name = name

    def get_name(self):
        return self.block_name
