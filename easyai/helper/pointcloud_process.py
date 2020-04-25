#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class PointCloudProcess():

    def __init__(self, dim):
        self.dim = dim

    def read_pointcloud(self, file_path):
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, self.dim)
        return point_cloud
