#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from torch_utility.torch_summary import summary
<<<<<<< HEAD
from model.utility.model_factory import ModelFactory
=======
from model.model_factory import ModelFactory
>>>>>>> e909c5896721a8536401dcdecbc87088f1d77f04


def main():
    print("process start...")
    model_factory = ModelFactory()
    model = model_factory.get_model("./cfg/yolov3.cfg")
    summary(model, [1, 3, 640, 352])
    print("process end!")


if __name__ == '__main__':
    main()
