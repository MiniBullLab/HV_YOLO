#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import torch.utils.data as data
from easyai.helper.dirProcess import DirProcess
from easyai.helper.imageProcess import ImageProcess


class SuperResolutionDataloader(data.Dataset):

    def __init__(self, input_dir, target_dir,
                 input_transform=None, target_transform=None):
        super().__init__()
        self.dir_process = DirProcess()
        self.image_process = ImageProcess()
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.input_files = list(self.dir_process.getDirFiles(input_dir))
        self.target_files = list(self.dir_process.getDirFiles(target_dir))

    def __getitem__(self, index):
        input_data = self.image_process.loadYUVImage(self.input_files[index])
        target_data = self.image_process.loadYUVImage(self.target_files[index])
        input = None
        target = None
        if self.input_transform:
            input = self.input_transform(input_data)
        else:
            input = torch.from_numpy(input)
        if self.target_transform:
            target = self.target_transform(target_data)
        else:
            target = torch.from_numpy(target)
        return input, target

    def __len__(self):
        return len(self.input_files)


def get_sr_dataloader(train_dir, val_dir, batch_size, num_workers=8,
                      shuffle=False, input_transform=None, target_transform=None):
    dataloader = SuperResolutionDataloader(train_dir, val_dir,
                                           input_transform=input_transform,
                                           target_transform=target_transform)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=shuffle)
    return result
