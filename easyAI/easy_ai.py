#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import inspect
from optparse import OptionParser
from easyAI.main import MainProcess


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program train model"

    parser.add_option("-t", "--task", dest="task_name",
                      type="string", default=None,
                      help="task name")

    parser.add_option("-g", "--gpu", dest="gpu_id",
                      type="int", default=0,
                      help="gpu id")

    parser.add_option("-i", "--trainPath", dest="trainPath",
                      metavar="PATH", type="string", default="./train.txt",
                      help="path to data config file")

    parser.add_option("-v", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    (options, args) = parser.parse_args()

    if options.trainPath:
        if not os.path.exists(options.trainPath):
            parser.error("Could not find the input train file")
        else:
            options.input_path = os.path.normpath(options.trainPath)
    else:
        parser.error("'trainPath' option is required to run this program")

    return options


def main():
    print("process start...")
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    cfg_path = os.path.join(dir_name, "./data/detection.cfg")
    pretrain_model_path = os.path.join(dir_name, "./data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    options = parse_arguments()
    my_main = MainProcess(options.task_name, cfg_path, options.gpu_id)
    my_main.process(options.trainPath, options.valPath, pretrain_model_path)
    print("process end!")


if __name__ == "__main__":
    main()
