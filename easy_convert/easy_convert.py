#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie


import os
import inspect
from optparse import OptionParser
from easy_convert.main_convert import model_convert


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program convert model"

    parser.add_option("-t", "--task", dest="task_name",
                      type="string", default=None,
                      help="task name")

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="onnx path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    return options


def main():
    print("process start...")
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    options = parse_arguments()
    model_convert(options.task_name, options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()
