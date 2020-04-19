import torch
from optparse import OptionParser
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.torch_to_onnx import TorchConvertOnnx
from easyai.config.utility.config_factory import ConfigFactory
from easyai.base_name.task_name import TaskName


def parse_arguments():

    parser = OptionParser()
    parser.description = "This program convert model to onnx"

    parser.add_option("-t", "--task", dest="task_name",
                      type="string", default=None,
                      help="task name")

    parser.add_option("-m", "--model", dest="model",
                      action="store", type="string", default=None,
                      help="model name or cfg file path")

    parser.add_option("-b", "--base_model", dest="base_model",
                      action="store", type="string", default=None,
                      help="base model name or cfg file path")

    parser.add_option("-p", "--weight_path", dest="weight_path",
                      metavar="PATH", type="string", default=None,
                      help="path to store weights")

    parser.add_option("-d", "--save_dir", dest="save_dir",
                      metavar="PATH", type="string", default=".",
                      help="save onnx dir")
    (options, args) = parser.parse_args()
    return options


class ModelConverter():

    def __init__(self, input_size=(352, 640)):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()
        self.converter = TorchConvertOnnx()
        self.input_size = input_size  # w * h

    def model_convert(self, model_path, weight_path, save_dir):
        input_x = torch.randn(1, 3, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        self.converter.set_save_dir(save_dir)
        model = self.model_factory.get_model(model_path)
        self.converter.torch2onnx(model, weight_path)

    def base_model_convert(self, base_model_path, weight_path, save_dir):
        input_x = torch.randn(1, 3, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        self.converter.set_save_dir(save_dir)
        model = self.backbone_factory.get_base_model(base_model_path)
        self.converter.torch2onnx(model, weight_path)


def main():
    pass


if __name__ == '__main__':
    options = parse_arguments()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(options.task_name, config_path=None)
    converter = ModelConverter(task_config.image_size)
    if options.model is not None:
        converter.model_convert(options.model, options.weight_path, options.save_dir)
    elif options.base_model is not None:
        converter.base_model_convert(options.base_model, options.weight_path, options.save_dir)
