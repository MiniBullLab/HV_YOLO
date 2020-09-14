#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper import DirProcess
from easyai.evaluation.classify_accuracy import ClassifyAccuracy
from easyai.helper.arguments_parse import ToolArgumentsParse


class OfflineClassifyEvaluation():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.evaluation = ClassifyAccuracy()

    def process(self, test_path, target_path):
        self.evaluation.clean_data()
        test_data_list = self.get_test_data(test_path)
        target_data_list = self.get_target_data(target_path)
        for image_name, class_index in test_data_list:
            for target_name, target_index in target_data_list:
                if image_name in target_name:
                    self.evaluation.result_eval(class_index, target_index)
                    break
        return self.evaluation.get_top1()

    def get_test_data(self, test_path):
        result = []
        for line_data in self.dirProcess.getFileData(test_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) == 2:
                # print(data_list[0])
                result.append((data_list[0], int(data_list[1])))
        return result

    def get_target_data(self, target_path):
        result = []
        for line_data in self.dirProcess.getFileData(target_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) == 2:
                # print(data_list[0])
                result.append((data_list[0], int(data_list[1])))
        return result


def main():
    print("start...")
    options = ToolArgumentsParse.test_path_parse()
    test = OfflineClassifyEvaluation()
    value = test.process(options.inputPath, options.targetPath)
    print("Classify Accuracy: {:.3f}%".format(value * 100))
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()


