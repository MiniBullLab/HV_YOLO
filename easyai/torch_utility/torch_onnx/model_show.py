import netron
from easyai.torch_utility.torch_onnx.torch_to_onnx import TorchConvertOnnx


class ModelShow():

    def __init__(self, save_dir="."):
        self.converter = TorchConvertOnnx()
        self.converter.set_save_dir(save_dir)

    def show_from_onnx(self, onnx_path):
        netron.start(onnx_path, port=9999, host='localhost')

    def set_input(self, input_torch):
        self.converter.set_input(input_torch)

    def show_from_model(self, model):
        save_onnx_path = self.converter.torch2onnx(model)
        self.show_from_onnx(save_onnx_path)
