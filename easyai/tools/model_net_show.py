import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.model_show import ModelShow
from easyai.helper.arguments_parse import ToolArgumentsParse


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
    options = ToolArgumentsParse.model_show_parse()
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
