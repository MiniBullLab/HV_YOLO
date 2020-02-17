#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from optparse import OptionParser
from easyAI.model_encrypt import ModelEncrypt


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is model encrypt"

    parser.add_option("-i", "--modelPath", dest="modelPath",
                      metavar="PATH", type="string", default="./model.bin",
                      help="path to model file")

    parser.add_option("-o", "--outputPath", dest="outputPath",
                      metavar="PATH", type="string", default="./model.bin",
                      help="path to model file")

    (options, args) = parser.parse_args()

    if options.modelPath:
        if not os.path.exists(options.modelPath):
            parser.error("Could not find the input train file")
        else:
            options.input_path = os.path.normpath(options.modelPath)
    else:
        parser.error("'modelPath' option is required to run this program")

    return options


def main():
    print("process start...")
    options = parse_arguments()
    process = ModelEncrypt()
    process.encrypt_file(options.modelPath, options.outputPath)
    print("process end!")


if __name__ == "__main__":
    main()