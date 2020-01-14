#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import inspect
from optparse import OptionParser
from easyAI.copy_image import CopyImage
from easyAI.runner.fgseg_train import FgSegV2Train
from easyAI.config import fgseg_config


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program train model"

    parser.add_option("-g", "--gpu", dest="gpu_id",
                      type="int", default=0,
                      help="gpu id")

    parser.add_option("-i", "--trainPath", dest="trainPath",
                      metavar="PATH", type="string", default="./train_val.txt",
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
    copy_process = CopyImage()
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    pretrain_model_path = os.path.join(dir_name, "./data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    options = parse_arguments()
    train_process = FgSegV2Train()
    train_process.train(options.trainPath, pretrain_model_path)
    copy_process.copy(options.trainPath, fgseg_config.save_image_dir)
    print("process end!")


if __name__ == "__main__":
    main()