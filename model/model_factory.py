import os
from .modelParse import ModelParse
from .modelName import ModelName
from .myModel import MyModel
from .MSRResNet import MSRResNet
from .MySRModel import MySRModel
#from .mobileV2FCN import MobileV2FCN

class ModelFactory():

    def __init__(self):
        self.modelParse = ModelParse()

    def getModelFromCfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.modelParse.readCfgFile(cfg_path)
        model = MyModel(model_define)
        model.setModelName(file_name)
        return model

    def getModelFromName(self, modelName):
        model = None
        if modelName == ModelName.MSRResNet:
            model = MSRResNet(in_nc=1, upscale_factor=3)
        elif model == ModelName.MySRModel:
            model = MySRModel(upscale_factor=3)
        # elif modelName == ModelName.MobileV2FCN:
        #     model = MobileV2FCN()
        return model
