import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + "/..")
from base_block.baseLayer import EmptyLayer, Upsample
from base_block.utilityBlock import ConvBNReLU
from base_model.mobilenetv2 import MobileNetV2
from base_model.baseModel import *
from .modelName import ModelName

class MobileV2FCN(BaseModel):
    """YOLOv3 object detection model"""

    def __init__(self, classNum = 2, freeze_bn=False):
        super().__init__()
        self.setModelName(ModelName.MobileV2FCN)

        self.basicModel = MobileNetV2(width_mult=1.0)
        basicModelChannels = self.basicModel.out_channels

        self.FCNStep1 = nn.Sequential(ConvBNReLU(basicModelChannels[4], basicModelChannels[4]//2, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//2, basicModelChannels[4], 3, 1, 1, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4], basicModelChannels[4]//2, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      Upsample(scale_factor = 2, mode='bilinear'))
        self.FCNSkipLayerStep1 = ConvBNReLU(basicModelChannels[3], basicModelChannels[4]//2, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d)
        self.FCNStep2 = nn.Sequential(ConvBNReLU(basicModelChannels[4], int(basicModelChannels[4]//4), 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//4, basicModelChannels[4]//2, 3, 1, 1, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//2, basicModelChannels[4]//4, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      Upsample(scale_factor=2, mode='bilinear'))
        self.FCNSkipLayerStep2 = ConvBNReLU(basicModelChannels[2], basicModelChannels[4]//4, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d)
        self.FCNStep3 = nn.Sequential(ConvBNReLU(basicModelChannels[4]//2, basicModelChannels[4]//8, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//8, basicModelChannels[4]//4, 3, 1, 1, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//4, basicModelChannels[4]//8, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      Upsample(scale_factor=2, mode='bilinear'))
        self.FCNSkipLayerStep3 = ConvBNReLU(basicModelChannels[1], basicModelChannels[4]//8, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d)
        self.FCNStep4 = nn.Sequential(ConvBNReLU(basicModelChannels[4]//4, basicModelChannels[4]//16, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//16, basicModelChannels[4]//8, 3, 1, 1, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      ConvBNReLU(basicModelChannels[4]//8, basicModelChannels[4]//16, 1, 1, 0, relu6=False,
                                        norm_layer=nn.BatchNorm2d),
                                      nn.Conv2d(int(basicModelChannels[4]//16), classNum, 1, 1, 0),
                                      Upsample(scale_factor=4, mode='bilinear'))

        self._init_weight()

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

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

    def forward(self, x):
        layer_outputs = []

        blocks = self.basicModel(x)

        x = self.FCNStep1(blocks[4])
        xSkip1 = self.FCNSkipLayerStep1(blocks[3])
        x = torch.cat([xSkip1, x], 1)
        x = self.FCNStep2(x)
        xSkip2 = self.FCNSkipLayerStep2(blocks[2])
        x = torch.cat([xSkip2, x], 1)
        x = self.FCNStep3(x)
        xSkip3 = self.FCNSkipLayerStep3(blocks[1])
        x = torch.cat([xSkip3, x], 1)
        x = self.FCNStep4(x)
        # print(x.shape)

        return x