#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.abstract_model import *


class BaseModel(AbstractModel):

    def __init__(self):
        super().__init__()
        self.lossList = []

    @abc.abstractmethod
    def create_loss(self, input_dict=None):
        pass
