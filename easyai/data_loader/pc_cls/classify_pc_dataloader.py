#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.utils.data as data
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.helper.pointcloud_process import PointCloudProcess
from easyai.data_loader.pc_cls.classify_pc_augment import ClassifyPointCloudAugment
from easyai.data_loader.pc_cls.classify_pc_process import ClassifyPointCloudProcess


class ClassifyPointCloudDataloader(data.Dataset):

    def __init__(self, train_path, dim, is_augment=False):
        self.is_augment = is_augment
        self.pointcloud_process = PointCloudProcess(dim=dim)
        self.classify_sample = ClassifySample(train_path)
        self.classify_sample.read_sample(flag=1)

        self.pc_augment = ClassifyPointCloudAugment()
        self.pc_process = ClassifyPointCloudProcess()

    def __getitem__(self, index):
        data_path, label = self.classify_sample.get_sample_path(index)
        point_cloud = self.pointcloud_process.read_pointcloud(data_path)
        if self.is_augment:
            point_cloud = self.pc_augment.augment(point_cloud)
        point_cloud = self.pc_process.normaliza_dataset(point_cloud)
        return point_cloud, label

    def __len__(self):
        return self.classify_sample.get_sample_count()


def get_classify_train_dataloader(train_path, dim, batch_size, is_augment=True, num_workers=8):
    dataloader = ClassifyPointCloudDataloader(train_path, dim=dim, is_augment=is_augment)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True, drop_last=True)
    return result


def get_classify_val_dataloader(val_path, dim, batch_size, num_workers=8):
    dataloader = ClassifyPointCloudDataloader(val_path, dim=dim, is_augment=False)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False, drop_last=True)
    return result
