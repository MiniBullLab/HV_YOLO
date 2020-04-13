#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess


class ClassifyPointCloudProcess(BaseDataSetProcess):

    def __init__(self):
        super().__init__()

    def normaliza_dataset(self, src_cloud):
        src_cloud[:, :3] = self.pc_normaliza(src_cloud[:, :3])
        result = self.numpy_transpose(src_cloud)
        result = self.numpy_to_torch(result, flag=0)
        return result

    def numpy_transpose(self, src_cloud):
        point_cloud = src_cloud.transpose(1, 0)
        return point_cloud

    def pc_normaliza(self, src_cloud):
        centroid = np.mean(src_cloud, axis=0)
        pc = src_cloud - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc
