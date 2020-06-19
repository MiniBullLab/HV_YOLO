#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
import caffe
import torch
from easy_convert.inference.det2d.keypoint_utility import *
from easy_convert.helper.dirProcess import DirProcess


class CaffeKeypointInference():

    def __init__(self, model_def, model_weights):
        self.dir_process = DirProcess()
        self.image_size = (640, 480)  # w, h
        self.output_node = 'layer30-conv'
        self.class_list = ['object']
        self.num_classes = len(self.class_list)
        self.thresh_conf = 0.1
        self.edges_corners = [[1, 2], [2, 4], [4, 3], [3, 1], [1, 5], [5, 6],
                              [6, 8], [8, 7], [7, 5], [7, 3], [8, 4], [6, 2]]
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)

    def keypoint_detect(self, image_dir):
        for img_path in self.dir_process.getDirFiles(image_dir):
            input_data = cv2.imread(img_path)
            input_data = cv2.resize(input_data, (self.image_size[0], self.image_size[1]))
            rgb_input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            image = rgb_input_data[:, :, ::-1].transpose(2, 0, 1)
            image = np.ascontiguousarray(image, dtype=np.float32)
            image /= 255.0
            image = image[np.newaxis, :]
            self.net.blobs['data'].data[...] = image

            # Forward pass
            self.net.forward()
            output = self.net.blobs[self.output_node].data
            output = torch.from_numpy(output).cuda()

            # Using confidence threshold, eliminate low-confidence predictions
            all_boxes = get_region_boxes(output, self.thresh_conf, self.num_classes)

            # Iterate through all images in the batch
            for i in range(output.size(0)):
                # For each image, get all the predictions
                boxes = all_boxes[i]

                # Iterate through each ground-truth object
                best_conf_est = -1

                # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
                for j in range(len(boxes)):
                    if (boxes[j][18] > best_conf_est):
                        box_pr = boxes[j]
                        best_conf_est = boxes[j][18]

                # Denormalize the corner predictions
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.image_size[0]
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.image_size[1]

                for edge in self.edges_corners:
                    cv2.line(input_data, (corners2D_pr[edge[0]][0], corners2D_pr[edge[0]][1]),
                             (corners2D_pr[edge[1]][0], corners2D_pr[edge[1]][1]), (0, 0, 255), 2)

                cv2.imshow("image", input_data)
            key = cv2.waitKey()
            if key == 1048603 or key == 27:
                break


if __name__ == "__main__":
    caffe.set_device(0)
    caffe.set_mode_gpu()

    imgdir = "/home/wfw/EDGE/3D-Pose/singleshotpose/LINEMOD/object/JPEGImages/"
    prototxt_file = '/home/wfw/EDGE/3D-Pose/singleshotpose/backup/object/yolo-pose_san.prototxt'
    weight_file = '/home/wfw/EDGE/3D-Pose/singleshotpose/backup/object/yolo-pose_san.caffemodel'
    test = CaffeKeypointInference(prototxt_file, weight_file)
    test.keypoint_detect(imgdir)