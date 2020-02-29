#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from optparse import OptionParser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is keras inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path")

    parser.add_option("-w", "--pb", dest="pb_path",
                      metavar="PATH", type="string", default=None,
                      help="tensorflow pb path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.pb_path:
        if not os.path.exists(options.pb_path):
            parser.error("Could not find the onnx file")
        else:
            options.pb_path = os.path.normpath(options.pb_path)
    else:
        parser.error("'pb_path' option is required to run this program")

    return options


class TensorflowInference():

    def __init__(self, pb_path):
        self.pb_path = pb_path

    def infer(self, input_path):
        sess = tf.Session()

        output_graph_def = tf.GraphDef()
        with open(self.pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(output_graph_def, name="")

        sess.run(tf.global_variables_initializer())
        self.print_op(sess)
        input_data = sess.graph.get_tensor_by_name("0:0")
        oput_data = sess.graph.get_tensor_by_name("Sigmoid:0")

        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        probs = sess.run(oput_data, feed_dict={input_data: img})
        # print(probs.shape)
        probs = probs.squeeze()
        threshold = 0.8
        probs[probs < threshold] = 0.
        probs[probs >= threshold] = 255.
        self.show(probs)

        sess.close()

    def print_op(self, sess):
        op = sess.graph.get_operations()
        for m in op:
            print(m.values())

    def show(self, result):
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(result)

        plt.title('Segmentation mask')
        plt.axis('off')
        plt.show()


def main():
    print("process start...")
    options = parse_arguments()
    inference = TensorflowInference(options.pb_path)
    inference.infer(options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()