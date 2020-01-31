#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.torch_utility.torch_summary import summary
from easyai.model.utility.model_factory import ModelFactory


def main():
    print("process start...")
    model_factory = ModelFactory()
    model = model_factory.get_model("./cfg/yolov3.cfg")
    summary(model, [1, 3, 640, 352])
    print("process end!")


if __name__ == '__main__':
    main()
