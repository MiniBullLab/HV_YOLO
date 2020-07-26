#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.base_model import *


class BaseDetectionModel(BaseModel):

    def __init__(self, data_channel, class_number):
        super().__init__(data_channel)
        self.class_number = class_number
