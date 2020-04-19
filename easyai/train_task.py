#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import ArgumentsParse
from easyai.tasks.cls.classify_train import ClassifyTrain
from easyai.tasks.det2d.detect2d_train import Detection2dTrain
from easyai.tasks.seg.segment_train import SegmentionTrain
from easyai.tools.model_to_onnx import ModelConverter
from easyai.base_name.task_name import TaskName


class TrainTask():

    def __init__(self, train_path, val_path, is_convert=False):
        self.train_path = train_path
        self.val_path = val_path
        self.is_convert = is_convert

    def classify_train(self, cfg_path, gpu_id, pretrain_model_path):
        classify_train = ClassifyTrain(cfg_path, gpu_id)
        classify_train.load_pretrain_model(pretrain_model_path)
        classify_train.train(self.train_path, self.val_path)
        if self.is_convert:
            converter = ModelConverter(classify_train.train_task_config.image_size)
            converter.model_convert(cfg_path, classify_train.train_task_config.best_weights_file,
                                    classify_train.train_task_config.snapshot_path)

    def detect2d_train(self, cfg_path, gpu_id, pretrain_model_path):
        detect2d_train = Detection2dTrain(cfg_path, gpu_id)
        detect2d_train.load_pretrain_model(pretrain_model_path)
        detect2d_train.train(self.train_path, self.val_path)
        if self.is_convert:
            converter = ModelConverter(detect2d_train.train_task_config.image_size)
            converter.model_convert(cfg_path, detect2d_train.train_task_config.best_weights_file,
                                    detect2d_train.train_task_config.snapshot_path)

    def segment_train(self, cfg_path, gpu_id, pretrain_model_path):
        segment_train = SegmentionTrain(cfg_path, gpu_id)
        segment_train.load_pretrain_model(pretrain_model_path)
        segment_train.train(self.train_path, self.val_path)
        if self.is_convert:
            converter = ModelConverter(segment_train.train_task_config.image_size)
            converter.model_convert(cfg_path, segment_train.train_task_config.best_weights_file,
                                    segment_train.train_task_config.snapshot_path)


def main():
    print("process start...")
    options = ArgumentsParse.train_input_parse()
    train_task = TrainTask(options.trainPath, options.valPath)
    if options.task_name == TaskName.Classify_Task:
        train_task.classify_train(options.cfg, 0, options.pretrainModel)
    elif options.task_name == TaskName.Detect2d_Task:
        train_task.detect2d_train(options.cfg, 0, options.pretrainModel)
    elif options.task_name == TaskName.Segment_Task:
        train_task.segment_train(options.cfg, 0, options.pretrainModel)
    print("process end!")


if __name__ == '__main__':
    main()
