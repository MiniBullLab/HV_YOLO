#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from optparse import OptionParser
import numpy as np
from keras.preprocessing import image
from easy_converter.keras_models.utility.keras_model_factory import KerasModelFactory
import matplotlib.pyplot as plt


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is keras inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path")

    parser.add_option("-w", "--h5", dest="h5_path",
                      metavar="PATH", type="string", default=None,
                      help="keras h5 path")

    parser.add_option("-m", "--model", dest="model_name",
                      type="string", default=None,
                      help="keras model name")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.h5_path:
        if not os.path.exists(options.h5_path):
            parser.error("Could not find the onnx file")
        else:
            options.h5_path = os.path.normpath(options.h5_path)
    else:
        parser.error("'onnx' option is required to run this program")

    return options


class KerasInference():

    def __init__(self, h5_model_path, model_name):
        self.model_factory = KerasModelFactory()
        self.h5_model_path = h5_model_path
        self.model_name = model_name

    def infer(self, input_path):
        model = self.model_factory.load_model(self.h5_model_path,
                                              self.model_name)
        print(input_path)
        x = self.load_image(input_path)
        probs = model.predict(x, batch_size=1, verbose=1)
        probs = probs.squeeze()
        threshold = 0.8
        probs[probs < threshold] = 0.
        probs[probs >= threshold] = 255.
        self.show(probs)

    def load_image(self, input_path):
        x = image.load_img(input_path)
        x = image.img_to_array(x)
        x /= 255.0
        x = np.expand_dims(x, axis=0)
        return x

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
    inference = KerasInference(options.h5_path, options.model_name)
    inference.infer(options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()
