#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.abstract_model import *


class BaseModel(AbstractModel):

    def __init__(self, data_channel):
        super().__init__()
        self.lossList = []
        self.data_channel = data_channel
        self.model_args = {"data_channel": data_channel}

    @abc.abstractmethod
    def create_loss(self, input_dict=None):
        pass
