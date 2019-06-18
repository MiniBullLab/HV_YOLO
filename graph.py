import netron
import torch
from torchvision.models import resnet18
from cfg.shufflenetv2_split import ShuffleNetV2Split
from modelsShuffleNet import *
import hiddenlayer as hl

img = torch.ones(1, 3, 640, 352)
model = ShuffleNetV2Split(net_size=1)

# y = model(img)
# for e_y in y:
#     print(e_y.shape)
# model = resnet18(True)
# torch.onnx.export(model, img, './weights/model.onnx')
# netron.start('./weights/model.onnx')

hl_graph = hl.build_graph(model, torch.zeros([1, 3, 640, 352]))
hl_graph.save(path="./model" , format="jpg")