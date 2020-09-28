#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc


class BaseOfflineEvaluation():

    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self, test_path, target_path):
        pass

    @abc.abstractmethod
    def print_evaluation(self, value):
        pass