import torch
from utility.model_summary import summary
from model.modelParse import ModelParse

modelParse = ModelParse()
model = modelParse.getModel("MobileV2FCN")
summary(model, [1, 3, 640, 352])