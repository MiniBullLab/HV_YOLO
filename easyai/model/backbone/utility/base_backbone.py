#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.abstract_model import *


class BaseBackbone(AbstractModel):

    def __init__(self, data_channel):
        super().__init__()
        self.data_channel = data_channel

    def get_outchannel_list(self):
        return self.block_out_channels
