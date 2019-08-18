import torch
from utility.model_summary import summary
from model.mobileV2FCN import *

model = MobileFCN("./cfg/mobileFCN.cfg", img_size=[640, 352])
summary(model, [1, 3, 640, 352])