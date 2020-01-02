#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

from easyAI.main import parse_arguments
from easyAI.main import MainProcess


def main():
    print("process start...")
    options = parse_arguments()
    my_main = MainProcess(options.task_name, options.cfg, options.gpu_id)
    my_main.process(options.trainPath, options.valPath, options.pretrainModel)
    print("process end!")


if __name__ == "__main__":
    main()
