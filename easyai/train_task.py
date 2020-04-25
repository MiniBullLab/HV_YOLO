#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import TaskArgumentsParse
from easyai.tasks.cls.classify_train import ClassifyTrain
from easyai.tasks.det2d.detect2d_train import Detection2dTrain
from easyai.tasks.seg.segment_train import SegmentionTrain
from easyai.tasks.sr.super_resolution_train import SuperResolutionTrain
from easyai.tools.model_to_onnx import ModelConverter
from easyai.base_name.task_name import TaskName


class TrainTask():

    def __init__(self, train_path, val_path, pretrain_model_path, is_convert=False):
        self.train_path = train_path
        self.val_path = val_path
        self.pretrain_model_path = pretrain_model_path
        self.is_convert = is_convert

    def classify_train(self, cfg_path, gpu_id, config_path):
        cls_train_task = ClassifyTrain(cfg_path, gpu_id, config_path)
        cls_train_task.load_pretrain_model(self.pretrain_model_path)
        cls_train_task.train(self.train_path, self.val_path)
        self.image_model_convert(cls_train_task, cfg_path)

    def detect2d_train(self, cfg_path, gpu_id, config_path):
        det2d_train = Detection2dTrain(cfg_path, gpu_id, config_path)
        det2d_train.load_pretrain_model(self.pretrain_model_path)
        det2d_train.train(self.train_path, self.val_path)
        self.image_model_convert(det2d_train, cfg_path)

    def segment_train(self, cfg_path, gpu_id, config_path):
        seg_train = SegmentionTrain(cfg_path, gpu_id, config_path)
        seg_train.load_pretrain_model(self.pretrain_model_path)
        seg_train.train(self.train_path, self.val_path)
        self.image_model_convert(seg_train, cfg_path)

    def super_resolution_train(self, cfg_path, gpu_id, config_path):
        sr_train = SuperResolutionTrain(cfg_path, gpu_id, config_path)
        sr_train.load_pretrain_model(self.pretrain_model_path)
        sr_train.train(self.train_path, self.val_path)
        self.image_model_convert(sr_train, cfg_path)

    def image_model_convert(self, train_task, cfg_path):
        if self.is_convert:
            converter = ModelConverter(train_task.train_task_config.image_size)
            converter.model_convert(cfg_path, train_task.train_task_config.best_weights_file,
                                    train_task.train_task_config.snapshot_path)


def main():
    print("process start...")
    options = TaskArgumentsParse.train_input_parse()
    train_task = TrainTask(options.trainPath, options.valPath, options.pretrainModel)
    if options.task_name == TaskName.Classify_Task:
        train_task.classify_train(options.model, 0, options.config_path)
    elif options.task_name == TaskName.Detect2d_Task:
        train_task.detect2d_train(options.model, 0, options.config_path)
    elif options.task_name == TaskName.Segment_Task:
        train_task.segment_train(options.model, 0, options.config_path)
    print("process end!")


if __name__ == '__main__':
    main()
