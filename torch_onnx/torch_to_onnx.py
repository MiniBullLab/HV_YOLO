import os.path
import torch
from torch import onnx
from torch_utility.torchModelProcess import TorchModelProcess


class TorchConvertOnnx():

    def __init__(self, channel=3, width=224, height=224):
        self.model_process = TorchModelProcess()
        self.input = torch.ones(1, channel, width, height)
        self.save_dir = "."

    def set_input(self, input_torch):
        self.input = input_torch

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def torch2onnx(self, model, weight_path=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if weight_path is not None:
            self.model_process.loadLatestModelWeight(weight_path, model)
        save_onnx_path = os.path.join(self.save_dir, "%s.onnx" % model.get_name())
        onnx.export(model, self.input, save_onnx_path)
        return save_onnx_path
