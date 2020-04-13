import torch
from optparse import OptionParser
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.model_show import ModelShow


def parse_arguments():

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

    parser.add_option("-p", "--pc_model", dest="pc_model",
                      action="store", type="string", default=None,
                      help="model name or cfg file path")

    parser.add_option("-c", "--pc_backbone", dest="pc_backbone",
                      action="store", type="string", default=None,
                      help="backbone name or cfg file path")

    (options, args) = parser.parse_args()
    return options


class ModelNetShow():

    def __init__(self):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()
        self.show_process = ModelShow()

    def model_show(self, model_path):
        input_x = torch.randn(1, 3, 224, 224)
        self.show_process.set_input(input_x)
        model = self.model_factory.get_model(model_path)
        self.show_process.show_from_model(model)

    def backbone_show(self, backbone_path):
        input_x = torch.randn(1, 3, 224, 224)
        self.show_process.set_input(input_x)
        model = self.backbone_factory.get_base_model(backbone_path)
        self.show_process.show_from_model(model)

    def onnx_show(self, onnx_path):
        input_x = torch.randn(1, 3, 640, 352)
        self.show_process.set_input(input_x)
        self.show_process.show_from_onnx(onnx_path)

    def pc_model_show(self, backbone_path):
        input_x = torch.randn(1, 3, 1024)
        self.show_process.set_input(input_x)
        model = self.model_factory.get_model(backbone_path)
        self.show_process.show_from_model(model)

    def pc_backbone_show(self, backbone_path):
        input_x = torch.randn(1, 3, 1024)
        self.show_process.set_input(input_x)
        model = self.backbone_factory.get_base_model(backbone_path)
        self.show_process.show_from_model(model)


def main():
    pass


if __name__ == '__main__':
    options = parse_arguments()
    show = ModelNetShow()
    if options.model is not None:
        show.model_show(options.model)
    elif options.backbone is not None:
        show.backbone_show(options.backbone)
    elif options.onnx_path is not None:
        show.onnx_show(options.onnx_path)
    elif options.pc_model is not None:
        show.pc_model_show(options.pc_model)
    elif options.pc_backbone is not None:
        show.pc_backbone_show(options.pc_backbone)
