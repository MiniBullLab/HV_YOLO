#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie


import os
import inspect
from optparse import OptionParser
from easyAI.convert import model_convert


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program train model"

    parser.add_option("-t", "--task", dest="task_name",
                      type="string", default=None,
                      help="task name")

    (options, args) = parser.parse_args()

    return options


def main():
    print("process start...")
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    options = parse_arguments()
    model_convert(options.task_name)
    print("process end!")


if __name__ == "__main__":
    main()
