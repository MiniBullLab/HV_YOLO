#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class BaseModel():

    def __init__(self):
        super().__init__()
        self.model_name = "None"

    def set_name(self, name):
        self.model_name = name

    def get_name(self):
        return self.model_name
