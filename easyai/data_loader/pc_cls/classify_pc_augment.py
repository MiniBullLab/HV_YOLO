#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class ClassifyPointCloudAugment():

    def __init__(self):
        self.is_augment_rotate = True
        self.is_augment_dropout = True
        self.is_augment_scale = True
        self.is_augment_shift = True
        self.rotation = (0, 30)

    def augment(self, src_cloud):
        if self.is_augment_rotate:
            src_cloud[:, :3] = self.random_rotate_point_cloud(src_cloud[:, :3])
        if self.is_augment_dropout:
            src_cloud = self.random_point_dropout(src_cloud)
        if self.is_augment_scale:
            src_cloud[:, :3] = self.random_scale_point_cloud(src_cloud[:, :3])
        if self.is_augment_shift:
            src_cloud[:, :3] = self.random_shift_point_cloud(src_cloud[:, :3])
        return src_cloud

    def random_rotate_point_cloud(self, src_cloud):
        """
        Rotate the point cloud along up direction with certain angle.
        :param src_cloud: Nx3 array, original batch of point clouds
        :return:  Nx3 array, rotated batch of point clouds
        """
        rotation_angle = np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(src_cloud, rotation_matrix)
        return rotated_data

    def random_point_dropout(self, src_cloud, max_dropout_ratio=0.875):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((src_cloud.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            src_cloud[drop_idx, :] = src_cloud[0, :]  # set to the first point
        return src_cloud

    def random_scale_point_cloud(self, src_cloud, scale_low=0.8, scale_high=1.25):
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                Nx3 array, original batch of point clouds
            Return:
                Nx3 array, scaled batch of point clouds
        """
        scales = np.random.uniform(scale_low, scale_high)
        src_cloud *= scales
        return src_cloud

    def random_shift_point_cloud(self, src_cloud, shift_range=0.1):
        """ Randomly shift point cloud. Shift is per point cloud.
            Input:
              Nx3 array, original batch of point clouds
            Return:
              Nx3 array, shifted batch of point clouds
        """
        shifts = np.random.uniform(-shift_range, shift_range, 3)
        src_cloud += shifts
        return src_cloud
