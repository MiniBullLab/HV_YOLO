#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class ModelParse():

    def __init__(self):
        pass

    def readCfgFile(self, cfgPath):
        model_defines = []
        file = open(cfgPath, 'r')
        lines = file.read().split('\n')
        lines = [x.strip() for x in lines if x.strip() and not x.startswith('#')]
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                model_defines.append({})
                model_defines[-1]['type'] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                value = value.strip()
                model_defines[-1][key.strip()] = value.strip()
        # print(model_defines)
        return model_defines

    def write_cfg_file(self, model_defines, cfg_path):
        with open(cfg_path, 'w') as f:
            for module_def in model_defines:
                f.write("[{}]\n".format(module_def['type']))
                for key, value in module_def.items():
                    if key != 'type':
                        f.write("{}={}\n".format(key, value))
                f.write("\n")
