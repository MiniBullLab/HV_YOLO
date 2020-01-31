import torch
from optparse import OptionParser
from easyai.model.backbone.utility.base_model_factory import BaseModelFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_onnx.model_show import ModelShow


def parse_arguments():

    parser = OptionParser()
    parser.description = "This program show model net"

    parser.add_option("-m", "--model", dest="model",
                      action="store", type="string", default=None,
                      help="model name or cfg file path")

    parser.add_option("-b", "--base_model", dest="base_model",
                      action="store", type="string", default=None,
                      help="base model name or cfg file path")

    parser.add_option("-o", "--onnx_path", dest="onnx_path",
                      action="store", type="string", default=None,
                      help="onnx file path")
    (options, args) = parser.parse_args()
    return options


class ModelNetShow():

    def __init__(self):
        self.base_model_factory = BaseModelFactory()
        self.model_factory = ModelFactory()
        self.show_process = ModelShow()

    def model_show(self, model_path):
        input_x = torch.randn(1, 3, 512, 440)
        self.show_process.set_input(input_x)
        model = self.model_factory.get_model(model_path)
        self.show_process.show_from_model(model)

    def base_model_show(self, base_model_path):
        input_x = torch.randn(1, 3, 224, 224)
        self.show_process.set_input(input_x)
        model = self.base_model_factory.get_base_model(base_model_path)
        self.show_process.show_from_model(model)

    def onnx_show(self, onnx_path):
        input_x = torch.randn(1, 3, 640, 352)
        self.show_process.set_input(input_x)
        self.show_process.show_from_onnx(onnx_path)


def main():
    pass


if __name__ == '__main__':
    options = parse_arguments()
    show = ModelNetShow()
    if options.model is not None:
        show.model_show(options.model)
    elif options.base_model is not None:
        show.base_model_show(options.base_model)
    elif options.onnx_path is not None:
        show.onnx_show(options.onnx_path)
