#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import inspect
from optparse import OptionParser
from easyai.tools.copy_image import CopyImage
from easyai.train_task import TrainTask
from easyai.config.utility.image_task_config import ImageTaskConfig
from easy_convert.convert_task import easy_model_convert


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

    parser.add_option("-c", "--config", dest="config_path",
                      metavar="PATH", type="string", default=None,
                      help="config path")

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
    options = parse_arguments()
    copy_process = CopyImage()
    config = ImageTaskConfig()
    train_task = TrainTask(options.trainPath, options.valPath, True)
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    if options.task_name.strip() == "ClassNet":
        cfg_path = os.path.join(dir_name, "./data/classnet.cfg")
        pretrain_model_path = os.path.join(dir_name, "./data/classnet.pt")
        train_task.classify_train(cfg_path, options.gpu_id, options.config_path, pretrain_model_path)
        save_image_dir = os.path.join(config.root_save_dir, "cls_img")
        copy_process.copy(options.trainPath, save_image_dir)
    elif options.task_name.strip() == "DeNET":
        pretrain_model_path = os.path.join(dir_name, "./data/detnet.pt")
        train_task.detect2d_train("YoloV3Det2d", options.gpu_id, options.config_path, pretrain_model_path)
        easy_model_convert(options.task_name, train_task.save_onnx_path)
        save_image_dir = os.path.join(config.root_save_dir, "det_img")
        copy_process.copy(options.trainPath, save_image_dir)
    elif options.task_name.strip() == "SegNET":
        pretrain_model_path = os.path.join(dir_name, "./data/segnet.pt")
        cfg_path = os.path.join(dir_name, "./data/segnet.cfg")
        train_task.segment_train(cfg_path, options.gpu_id, options.config_path, pretrain_model_path)
        save_image_dir = os.path.join(config.root_save_dir, "seg_img")
        copy_process.copy(options.trainPath, save_image_dir)
    else:
        print("input task error!")
    print("process end!")


if __name__ == "__main__":
    main()
