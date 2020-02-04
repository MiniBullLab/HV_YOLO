#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.abstract_model import *


class BaseBackbone(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_outchannel_list(self):
        return self.out_channels

    def print_block_name(self):
        for key in self._modules.keys():
            print(key)
