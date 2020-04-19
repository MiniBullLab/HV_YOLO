#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class ModelParse():

    def __init__(self):
        pass

    def readCfgFile(self, cfgPath):
        modelDefine = []
        file = open(cfgPath, 'r')
        lines = file.read().split('\n')
        lines = [x.strip() for x in lines if x.strip() and not x.startswith('#')]
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                modelDefine.append({})
                modelDefine[-1]['type'] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                value = value.strip()
                modelDefine[-1][key.strip()] = value.strip()
        #print(modelDefine)
        return modelDefine
