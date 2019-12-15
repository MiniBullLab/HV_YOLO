import os
from optparse import OptionParser


class ArgumentsParse():

    def __init__(self):
        pass

    @classmethod
    def test_input_parse(cls):
        parser = OptionParser()
        parser.description = "This program test model"

        parser.add_option("-t", "--task", dest="task_name",
                          metavar="PATH", type="string", default="",
                          help="path to data config file")

        parser.add_option("-i", "--valPath", dest="valPath",
                          metavar="PATH", type="string", default="./val.txt",
                          help="path to data config file")

        parser.add_option("-c", "--cfg", dest="cfg",
                          metavar="PATH", type="string", default="cfg/cifar100.cfg",
                          help="cfg file path")

        parser.add_option("-w", "--weights", dest="weights",
                          metavar="PATH", type="string", default="weights/latest.pt",
                          help="path to store weights")

        (options, args) = parser.parse_args()

        if options.valPath:
            if not os.path.exists(options.valPath):
                parser.error("Could not find the input val file")
            else:
                options.input_path = os.path.normpath(options.valPath)
        else:
            parser.error("'valPath' option is required to run this program")

        return options

    @classmethod
    def train_input_parse(cls):
        parser = OptionParser()
        parser.description = "This program train model"

        parser.add_option("-t", "--task", dest="task_name",
                          metavar="PATH", type="string", default="",
                          help="path to data config file")

        parser.add_option("-i", "--trainPath", dest="trainPath",
                          metavar="PATH", type="string", default="./train.txt",
                          help="path to data config file")

        parser.add_option("-v", "--valPath", dest="valPath",
                          metavar="PATH", type="string", default="./val.txt",
                          help="path to data config file")

        parser.add_option("-c", "--cfg", dest="cfg",
                          metavar="PATH", type="string", default="cfg/cifar100.cfg",
                          help="cfg file path")

        parser.add_option("-p", "--pretrainModel", dest="pretrainModel",
                          metavar="PATH", type="string", default="weights/pretrain.pt",
                          help="path to store weights")

        (options, args) = parser.parse_args()

        if options.trainPath:
            if not os.path.exists(options.trainPath):
                parser.error("Could not find the input train file")
            else:
                options.input_path = os.path.normpath(options.trainPath)
        else:
            parser.error("'trainPath' option is required to run this program")

        return options

    @classmethod
    def parse_arguments(cls):

        parser = OptionParser()
        parser.description = "This program task"

        parser.add_option("-t", "--task", dest="task_name",
                          metavar="PATH", type="string", default="",
                          help="path to data config file")

        parser.add_option("-i", "--input", dest="inputPath",
                          metavar="PATH", type="string", default=None,
                          help="images path or video path")

        parser.add_option("-c", "--cfg", dest="cfg",
                          metavar="PATH", type="string", default="cfg/yolov3.cfg",
                          help="cfg file path")

        parser.add_option("-w", "--weights", dest="weights",
                          metavar="PATH", type="string", default="weights/latest.pt",
                          help="path to store weights")

        (options, args) = parser.parse_args()

        if options.inputPath:
            if not os.path.exists(options.inputPath):
                parser.error("Could not find the input file")
            else:
                options.input_path = os.path.normpath(options.inputPath)
        else:
            parser.error("'input' option is required to run this program")

        return options
