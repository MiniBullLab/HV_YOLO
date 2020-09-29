#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import cv2
import numpy as np
import random
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.helper.json_process import JsonProcess
from easyai.helper import ImageProcess
from easyai.base_name.task_name import TaskName
from easyai.config.utility.config_factory import ConfigFactory


class CreateDetectionAnchors():

    def __init__(self, train_path, config_path):
        self.config_factory = ConfigFactory()
        self.task_config = self.config_factory.get_config(TaskName.Detect2d_Task, config_path)

        # image & label process
        self.json_process = JsonProcess()
        self.image_process = ImageProcess()

        self.detection_sample = DetectionSample(train_path,
                                                self.task_config.detect2d_class)
        self.detection_sample.read_sample()

        self.dataset_process = DetectionDataSetProcess(1, 0)

    def get_anchors(self, number):
        wh_numpy = self.get_width_height()
        indices = [random.randrange(wh_numpy.shape[0]) for _ in range(number)]
        centroids = wh_numpy[indices]
        self.kmeans(wh_numpy, centroids)

    def get_width_height(self):
        count = self.detection_sample.get_sample_count()
        result = []
        print("count: {}".format(count))
        for index in range(count):
            img_path, label_path = self.detection_sample.get_sample_path(index)
            print("Loading : {}-{}".format(index, img_path))
            _, rgb_image = self.image_process.readRgbImage(img_path)
            _, boxes = self.json_process.parse_rect_data(label_path)
            rgb_image, labels = self.dataset_process.resize_dataset(rgb_image,
                                                                    self.task_config.image_size,
                                                                    boxes,
                                                                    self.task_config.detect2d_class)

            temp = np.zeros((len(labels), 2), dtype=np.float32)
            for index, object in enumerate(labels):
                temp[index, :] = np.array([object.width(), object.height()])
            result.append(temp)

        return np.concatenate(result, axis=0)

    def kmeans(self, wh_numpy, centroids):

        num_boxes = wh_numpy.shape[0]
        k, dim = centroids.shape
        prev_assignments = np.ones(num_boxes) * (-1)
        iteration = 0
        old_dists = np.zeros((num_boxes, k))

        while True:
            dists = []
            iteration += 1
            for i in range(num_boxes):
                d = 1 - self.compute_iou(wh_numpy[i], centroids)
                dists.append(d)
            dists = np.array(dists)  # dists.shape = (num_boxes, k)

            print("iteration {}: distance = {}".format(iteration, np.sum(np.abs(old_dists - dists))))

            # assign samples to centroids
            assignments = np.argmin(dists, axis=1)

            if (assignments == prev_assignments).all():
                self.anchor_visual(centroids)
                return

            # calculate new centroids
            centroid_sums = np.zeros((k, dim), np.float)
            for i in range(num_boxes):
                centroid_sums[assignments[i]] += wh_numpy[i]
            for j in range(k):
                centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

            prev_assignments = assignments.copy()
            old_dists = dists.copy()

    def compute_iou(self, x, centroids):
        similarities = []
        for centroid in centroids:
            centroid_w, centroid_h = centroid
            width, height = x
            if centroid_w >= width and centroid_h >= height:
                similarity = width * height / (centroid_w * centroid_h)
            elif centroid_w >= width and centroid_h <= height:
                similarity = width * centroid_h / (width * height + (centroid_w - width) * centroid_h)
            elif centroid_w <= width and centroid_h >= height:
                similarity = centroid_w * height / (width * height + centroid_w * (centroid_h - height))
            else:  # means both w,h are bigger than c_w and c_h respectively
                similarity = (centroid_w * centroid_h) / (width * height)
            similarities.append(similarity)  # will become (k,) shape
        return np.array(similarities)

    def anchor_visual(self, centroids):
        anchors = centroids.copy()

        widths = anchors[:, 0]
        sorted_indices = np.argsort(widths)

        print('Anchors = ', anchors[sorted_indices])

        # draw all anchors
        img = np.zeros([300, 300, 3])
        for i in range(anchors.shape[0]):
            cv2.rectangle(img, (int(150 - anchors[i][0] / 2.0), int(150 - anchors[i][1] / 2.0)),
                          (int(150 + anchors[i][0] / 2.0), int(150 + anchors[i][1] / 2.0)), (0, 0, 255), 1)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(img.shape[1] * 3.0), int(img.shape[0] * 3.0))
        cv2.imshow("image", img)
        cv2.waitKey()


def test():
    print("start...")
    test = CreateDetectionAnchors("/home/wfw/data/VOCdevkit/BerkeleyDet/ImageSets/train.txt",
                                  "/home/wfw/EDGE/HV_YOLO/log/config/detection2d_config.json")
    test.get_anchors(9)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()
