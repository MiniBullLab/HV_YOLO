#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class KeyPointAccuracy():

    def __init__(self, points_count, class_names):
        self.points_count = points_count
        self.class_names = class_names
        self.pixel_threshold = 5
        self.class_errors = {}

    def reset(self):
        self.class_errors = {}
        for index in range(len(self.class_names)):
            self.class_errors[index] = []

    def eval(self, result_objects, numpy_targets):
        numpy_outputs = self.objects_to_numpy(result_objects)
        class_index_list = np.unique(numpy_outputs[:, -1])

        for class_index in class_index_list:
            class_objects = numpy_outputs[numpy_outputs[:, -1] == class_index]
            target_objects = numpy_targets[numpy_targets[:, 0] == class_index]
            object_points = class_objects[:, 1:]
            target_points = target_objects[:, 1:]
            errors_object = []
            for gt_data in target_points:
                gt_points = np.array(np.reshape(gt_data, [self.points_count, 2]), dtype='float32')
                min_dist = 1e5
                for pr_data in object_points:
                    pr_points = np.array(np.reshape(pr_data, [self.points_count, 2]), dtype='float32')
                    points_norm = np.linalg.norm(pr_points - gt_points, axis=1)
                    points_dist = np.mean(points_norm)
                    if points_dist < min_dist:
                        min_dist = points_dist
                errors_object.append(min_dist)
            self.class_errors[class_index].extend(errors_object)

    def get_accuracy(self):
        all_accuracy = []
        for key, value in self.class_errors.items():
            class_acc = len(np.where(np.array(value) <= self.pixel_threshold)) \
                        * 100. / (len(value) + 1e-5)
            all_accuracy.append(class_acc)
        self.print_evaluation(all_accuracy)
        return np.mean(all_accuracy), all_accuracy

    def print_evaluation(self, all_accuracy):
        print('Mean accuracy = {:.4f}'.format(np.mean(all_accuracy)))
        print('~~~~~~~~')
        print('Results:')
        for i, acc in enumerate(all_accuracy):
            print(self.class_names[i] + ': ' + '{:.3f}'.format(acc))
        print('~~~~~~~~')

    def objects_to_numpy(self, input_objects):
        result = np.zeros((len(input_objects), self.points_count * 2 + 1), dtype=np.float32)
        for index, temp_object in enumerate(input_objects):
            key_points = temp_object.get_key_points()
            points = list()
            points.append(temp_object.classIndex)
            for index in range(0, self.points_count, 2):
                points.append(key_points[index].x)
                points.append(key_points[index].y)
            result[index, :] = np.array(points)
        return result
