#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2
import caffe
import numpy as np
from easy_convert.inference.det2d.keypoint2d_process import KeyPoint2dProcess
from easy_convert.helper.dirProcess import DirProcess
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is caffe key_point2d inference"

    parser.add_option("-i", "--image_dir", dest="image_dir",
                      type="string", default=None,
                      help="image dir")

    parser.add_option("-p", "--prototxt", dest="prototxt_path",
                      metavar="PATH", type="string", default=None,
                      help="caffe prototxt path")

    parser.add_option("-m", "--caffe_model", dest="caffe_model_path",
                      type="string", default=None,
                      help="caffe model path")

    parser.add_option("-o", "--output_node", dest="output_node",
                      type="string", default=None,
                      help="output node name")

    (options, args) = parser.parse_args()

    return options


class CaffeKeypointInference():

    def __init__(self, model_def, model_weights, output_node):
        self.dir_process = DirProcess()
        self.image_size = (640, 480)  # w, h
        self.output_node = output_node
        self.class_list = ['object']
        self.num_classes = len(self.class_list)
        self.thresh_conf = 0.1
        self.edges_corners = [[1, 2], [2, 4], [4, 3], [3, 1], [1, 5], [5, 6],
                              [6, 8], [8, 7], [7, 5], [7, 3], [8, 4], [6, 2]]
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.keypoint_inference = KeyPoint2dProcess(self.image_size, self.thresh_conf, self.num_classes)

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
            corners2D = self.keypoint_inference.postprocess(output)

            for edge in self.edges_corners:
                cv2.line(input_data, (corners2D[edge[0]][0], corners2D[edge[0]][1]),
                         (corners2D[edge[1]][0], corners2D[edge[1]][1]), (0, 0, 255), 2)

                cv2.imshow("image", input_data)
            key = cv2.waitKey()
            if key == 1048603 or key == 27:
                break


if __name__ == "__main__":
    caffe.set_device(0)
    caffe.set_mode_gpu()

    print("process start...")
    options = parse_arguments()
    test = CaffeKeypointInference(options.prototxt_path,
                                  options.caffe_model_path,
                                  options.output_node)
    test.keypoint_detect(options.image_dir)
    print("process end!")
