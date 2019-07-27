# Initialize model
import netron
import torch
from model.mobileV2FCN import *

def torch2onnx(model, input): # torchArch,

    torch.onnx.export(model, input, "./mobileSeg.onnx")
    netron.start("./mobileSeg.onnx")

def main():
    # model = MobileFCN("./cfg/mobileFCN.cfg", img_size=[640, 352])
    model = MobileNetV2(width_mult=1.0)
    print(model.state_dict())
    x = torch.ones(1, 3, 640, 352)
    torch2onnx(model, x)

if __name__=="__main__":
    main()

