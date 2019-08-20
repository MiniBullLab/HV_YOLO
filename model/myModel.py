import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.baseModel import *
from collections import defaultdict
from base_model.mobilenetv2 import MobileNetV2
from base_model.shufflenetv2 import ShuffleNetV2
from .createModelList import *

class MyModel(BaseModel):

    def __init__(self, modelDefine, freezeBn=False):
        super().__init__()
        self.createTaskList = CreateModuleList()
        self.modelDefine = modelDefine
        self.freezeBn = freezeBn
        self.basicModel = self.creatBaseModel()
        self.taskModules = self.createTask()
        self.lossList = self.createLoss()
        self.setFreezeBn()
        self._init_weight()

    def creatBaseModel(self):
        result = None
        baseNet = self.modelDefine[0]
        if baseNet["type"] == "baseNet":
            baseNetName = baseNet["name"]
            self.modelDefine.pop(0)
            if baseNetName == "ShuffleNetV2":
                result = ShuffleNetV2(net_size=1)
            elif baseNetName == "MobileNetV2":
                result = MobileNetV2(width_mult=1.0)
        return result

    def createTask(self):
        moduleList = nn.ModuleList()
        if self.basicModel:
            outChannels = self.basicModel.out_channels
            moduleList = self.createTaskList.getModuleList(outChannels, self.modelDefine)
        return moduleList

    def createLoss(self):
        # yoloLoss
        lossResult = []
        for m in self.taskModules:
            for layer in m:
                if isinstance(layer, YoloLoss):
                    lossResult.append(layer)
        return lossResult

    def setFreezeBn(self):
        for m in self.modules():
            if self.freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        self.losses = defaultdict(float)
        layer_outputs = []
        output = []

        blocks = self.basicModel(x)
        x = blocks[-1]

        for i, (module_def, module) in enumerate(zip(self.modelDefine, self.taskModules)):
            if module_def['type'] in [ModuleType.Convolutional, ModuleType.Upsample, ModuleType.Maxpool]:
                x = module(x)
            elif module_def['type'] == ModuleType.Route:
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] if i < 0 else blocks[i] for i in layer_i], 1)
            elif module_def['type'] == ModuleType.Shortcut:
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == ModuleType.Yolo:
                output.append(x)

            layer_outputs.append(x)

        return output

    def _init_weight(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)