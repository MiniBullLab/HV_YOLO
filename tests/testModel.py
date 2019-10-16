import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from model.model_factory import ModelFactory
from drawing.modelNetShow import ModelNetShow

def test_SR():
    model_factory = ModelFactory()
    input = torch.randn(1, 1, 72, 72)
    modelNetShow = ModelNetShow()
    model = model_factory.getModelFromName("MSRResNet")
    modelNetShow.setInput(input)
    modelNetShow.setSaveDir("../onnx")
    modelNetShow.showNet(model)

def main():
    model_factory = ModelFactory()
    model = model_factory.getModelFromCfg('../cfg/shufflenetV2-0.5_spp_BerkeleyAll.cfg')
    print(model)

if __name__ == '__main__':
    #main()
    test_SR()