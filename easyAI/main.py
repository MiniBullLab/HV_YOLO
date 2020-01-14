#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from optparse import OptionParser
from easyAI.runner.detectTrain import DetectionTrain
from easyAI.tools.model_to_onnx import ModelConverter

from easyAI.converter.keras_models.model_name import KerasModelName
from easyAI.runner.fgseg_train import FgSegV2Train
from easyAI.tools.keras_convert_tensorflow import KerasConvertTensorflow

from easyAI.config import base_config
from easyAI.config import detectConfig
from easyAI.config import fgseg_config

__all__ = ['MainProcess']


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

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/cifar100.cfg",
                      help="cfg file path")

    parser.add_option("-p", "--pretrainModel", dest="pretrainModel",
                      metavar="PATH", type="string", default="weights/pretrain.pt",
                      help="path to store weights")

    (options, args) = parser.parse_args()

    if options.trainPath:
        if not os.path.exists(options.trainPath):
            parser.error("Could not find the input train file")
        else:
            options.input_path = os.path.normpath(options.trainPath)
    else:
        parser.error("'trainPath' option is required to run this program")

    return options


class MainProcess():

    def __init__(self, task_name, cfg_path, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if not os.path.exists(base_config.root_save_dir):
            os.makedirs(base_config.root_save_dir, exist_ok=True)
        self.task_name = task_name.strip()
        self.gpu_id = gpu_id
        self.cfg_path = cfg_path
        self.save_onnx_path = None

    def process_train(self, train_path, val_path=None, vgg_weights_path=None):
        if self.task_name == "DeNET":
            if os.path.exists(self.cfg_path):
                detect_train = DetectionTrain(self.cfg_path, self.gpu_id)
            else:
                detect_train = None
            detect_train.train(train_path, val_path)
            onnx_converter = ModelConverter()
            self.save_onnx_path = onnx_converter.model_convert(self.cfg_path,
                                                               detectConfig.best_weights_file,
                                                               detectConfig.snapshotPath)
        elif self.task_name == "SegNET":
            seg_train = FgSegV2Train()
            seg_train.train(train_path, vgg_weights_path)
            pb_converter = KerasConvertTensorflow(fgseg_config.best_weights_file,
                                                  KerasModelName.MyFgSegNetV2)
            pb_converter.keras_convert_tensorflow()
        else:
            print("error input")


def main():
    print("process start...")
    options = parse_arguments()
    my_main = MainProcess(options.task_name, options.cfg, options.gpu_id)
    my_main.process_train(options.trainPath, options.valPath, options.pretrainModel)
    print("process end!")


if __name__ == "__main__":
    main()
