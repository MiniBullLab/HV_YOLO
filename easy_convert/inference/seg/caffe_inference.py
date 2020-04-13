#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from optparse import OptionParser
import caffe
import cv2
import numpy as np
import matplotlib.pyplot as plt
from easy_convert.helper.image_dataset_process import ImageDataSetProcess


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is keras inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path")

    parser.add_option("-p", "--proto", dest="proto_path",
                      metavar="PATH", type="string", default=None,
                      help="prototxt path")

    parser.add_option("-w", "--weights", dest="weights_path",
                      metavar="PATH", type="string", default=None,
                      help="caffe weights path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.proto_path:
        if not os.path.exists(options.proto_path):
            parser.error("Could not find the prototxt file")
        else:
            options.proto_path = os.path.normpath(options.proto_path)
    else:
        parser.error("'proto' option is required to run this program")

    if options.weights_path:
        if not os.path.exists(options.weights_path):
            parser.error("Could not find the weights file")
        else:
            options.weights_path = os.path.normpath(options.weights_path)
    else:
        parser.error("'weights' option is required to run this program")

    return options


class CaffeInference():

    def __init__(self, prototxt_path, weights_path):
        self.dataset_process = ImageDataSetProcess()
        self.prototxt_path = prototxt_path
        self.weights_path = weights_path
        self.image_size = (640, 352)

    def infer(self, input_path):
        net = caffe.Net(self.prototxt_path, self.weights_path, caffe.TEST)
        input_data = cv2.imread(input_path)
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = self.preprocess(input_data, self.image_size)
        net.blobs['0'].data[...] = input_data
        net.forward()
        prob = net.blobs['467'].data[0]
        prediction = self.postprocess(prob)
        self.show(prediction)

    def preprocess(self, input_data, image_size):
        image, _, _ = self.dataset_process.image_resize_square(input_data,
                                                               image_size,
                                                               color=(127.5, 127.5, 127.5))
        # Normalize RGB
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255.0
        image = image[np.newaxis, :]
        return image

    def postprocess(self, prob):
        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
        return prediction

    def show(self, result):
        plt.imshow(result)
        plt.title('Segmentation mask')
        plt.axis('off')
        plt.show()


def main():
    print("process start...")
    options = parse_arguments()
    inference = CaffeInference(options.proto_path, options.weights_path)
    inference.infer(options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()


