import os
from optparse import OptionParser


class TaskArgumentsParse():

    def __init__(self):
        pass

    @classmethod
    def test_input_parse(cls):
        parser = OptionParser()
        parser.description = "This program test model"

        parser.add_option("-t", "--task", dest="task_name",
                          type="string", default=None,
                          help="task name")

        parser.add_option("-i", "--valPath", dest="valPath",
                          metavar="PATH", type="string", default="./val.txt",
                          help="path to data config file")

        parser.add_option("-m", "--model", dest="model",
                          metavar="PATH", type="string", default="cfg/conv_block.cfg",
                          help="cfg file path or model name")

        parser.add_option("-w", "--weights", dest="weights",
                          metavar="PATH", type="string", default="weights/latest.pt",
                          help="path to store weights")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

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
                          type="string", default=None,
                          help="task name")

        parser.add_option("-i", "--trainPath", dest="trainPath",
                          metavar="PATH", type="string", default="./train.txt",
                          help="path to data config file")

        parser.add_option("-v", "--valPath", dest="valPath",
                          metavar="PATH", type="string", default=None,
                          help="path to data config file")

        parser.add_option("-m", "--model", dest="model",
                          metavar="PATH", type="string", default="cfg/conv_block.cfg",
                          help="cfg file path or model name")

        parser.add_option("-p", "--pretrainModel", dest="pretrainModel",
                          metavar="PATH", type="string", default=None,
                          help="path to store weights")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

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
    def inference_parse_arguments(cls):

        parser = OptionParser()
        parser.description = "This program task"

        parser.add_option("-t", "--task", dest="task_name",
                          type="string", default=None,
                          help="task name")

        parser.add_option("-i", "--input", dest="inputPath",
                          metavar="PATH", type="string", default=None,
                          help="images path or video path")

        parser.add_option("-m", "--model", dest="model",
                          metavar="PATH", type="string", default="cfg/conv_block.cfg",
                          help="cfg file path or model name")

        parser.add_option("-w", "--weights", dest="weights",
                          metavar="PATH", type="string", default="weights/latest.pt",
                          help="path to store weights")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

        (options, args) = parser.parse_args()

        if options.inputPath:
            if not os.path.exists(options.inputPath):
                parser.error("Could not find the input file")
            else:
                options.input_path = os.path.normpath(options.inputPath)
        else:
            parser.error("'input' option is required to run this program")

        return options


class ToolArgumentsParse():

    def __init__(self):
        pass

    @classmethod
    def dir_path_parse(cls):
        parser = OptionParser()
        parser.description = "This program"

        parser.add_option("-i", "--input", dest="inputPath",
                          metavar="PATH", type="string", default=None,
                          help="images dir")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

        (options, args) = parser.parse_args()

        if options.inputPath:
            if not os.path.exists(options.inputPath):
                parser.error("Could not find the input file")
            else:
                options.inputPath = os.path.normpath(options.inputPath)
        else:
            parser.error("'input' option is required to run this program")

        return options

    @classmethod
    def test_path_parse(cls):
        parser = OptionParser()
        parser.description = "This program test"

        parser.add_option("-i", "--input", dest="inputPath",
                          metavar="PATH", type="string", default=None,
                          help="test images")

        parser.add_option("-t", "--target", dest="targetPath",
                          metavar="PATH", type="string", default=None,
                          help="target images")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

        (options, args) = parser.parse_args()

        if options.inputPath:
            if not os.path.exists(options.inputPath):
                parser.error("Could not find the input file")
            else:
                options.inputPath = os.path.normpath(options.inputPath)
        else:
            parser.error("'input' option is required to run this program")

        if options.targetPath:
            if not os.path.exists(options.targetPath):
                parser.error("Could not find the input file")
            else:
                options.targetPath = os.path.normpath(options.targetPath)
        else:
            parser.error("'target' option is required to run this program")

        return options

    @classmethod
    def images_path_parse(cls):
        parser = OptionParser()
        parser.description = "This program"

        parser.add_option("-t", "--task", dest="task_name",
                          type="string", default=None,
                          help="task name")

        parser.add_option("-i", "--input", dest="inputPath",
                          metavar="PATH", type="string", default=None,
                          help="images text")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

        (options, args) = parser.parse_args()

        if options.inputPath:
            if not os.path.exists(options.inputPath):
                parser.error("Could not find the input file")
            else:
                options.input_path = os.path.normpath(options.inputPath)
        else:
            parser.error("'input' option is required to run this program")

        return options

    @classmethod
    def model_show_parse(cls):
        parser = OptionParser()
        parser.description = "This program show model net"

        parser.add_option("-m", "--model", dest="model",
                          action="store", type="string", default=None,
                          help="model name or cfg file path")

        parser.add_option("-b", "--backbone", dest="backbone",
                          action="store", type="string", default=None,
                          help="backbone name or cfg file path")

        parser.add_option("-o", "--onnx_path", dest="onnx_path",
                          action="store", type="string", default=None,
                          help="onnx file path")

        (options, args) = parser.parse_args()
        return options

    @classmethod
    def model_convert_parse(cls):
        parser = OptionParser()
        parser.description = "This program convert model to onnx"

        parser.add_option("-t", "--task", dest="task_name",
                          type="string", default=None,
                          help="task name")

        parser.add_option("-m", "--model", dest="model",
                          action="store", type="string", default=None,
                          help="model name or cfg file path")

        parser.add_option("-b", "--backbone", dest="backbone",
                          action="store", type="string", default=None,
                          help="backbone name or cfg file path")

        parser.add_option("-p", "--weight_path", dest="weight_path",
                          metavar="PATH", type="string", default=None,
                          help="path to store weights")

        parser.add_option("-d", "--save_dir", dest="save_dir",
                          metavar="PATH", type="string", default=".",
                          help="save onnx dir")
        (options, args) = parser.parse_args()
        return options

    @classmethod
    def show_lr_parse(cls):
        parser = OptionParser()
        parser.description = "This program convert model to onnx"

        parser.add_option("-t", "--task", dest="task_name",
                          type="string", default=None,
                          help="task name")

        parser.add_option("-e", "--epoch_iteration", dest="epoch_iteration",
                          action="store", type="int", default=4000,
                          help="model name or cfg file path")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")
