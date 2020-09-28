#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.torch_utility.torch_summary import summary
from easyai.model.utility.model_factory import ModelFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


def main():
    print("process start...")
    options = ToolArgumentsParse.model_parse()
    model_factory = ModelFactory()
    model_config = {"type": options.model,
                    "data_channel": 3}
    model = model_factory.get_model(model_config)
    summary(model, [1, 3, 640, 352])
    print("process end!")


if __name__ == '__main__':
    main()
